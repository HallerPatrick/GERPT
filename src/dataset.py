import os
from typing import List, Tuple

import flair
from flair.embeddings.token import FlairEmbeddings
import nltk

from numba import jit

import pytorch_lightning as pl
import torch
import numpy as np
import dask
from dask.diagnostics import ProgressBar

from codetiming import Timer
from datasets import load_dataset as ld

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm.contrib.concurrent import process_map
from humanfriendly import format_timespan

from . import SERVER_CACHE, USE_CACHE
from .dictionary import Dictionary
from .data import batchify, local_dataset_mapper


def batch_collate(batch):
    # TODO: Little akward that we get two lists where each list contains 2xbatch for source and target

    source = []
    target = []
    for mini_batch in batch:
        source.append(mini_batch[0])
        target.append(mini_batch[1])

    # Pre-tokenized dataset is int16 for memory reasons
    source_tensor = torch.tensor(np.stack(source, axis=2), dtype=torch.int64)
    target_tensor = torch.tensor(np.stack(target, axis=2), dtype=torch.int64)
    return source_tensor, target_tensor


class TextDataset(Dataset):
    def __init__(self, ds, batch_size, bptt_size, pad_tokens) -> None:
        self.bptt = bptt_size
        self.batch_size = batch_size
        self.inputs, self.nbatch = batchify(ds["source"], batch_size, bptt_size)
        self.target, _ = batchify(ds["target"], batch_size, bptt_size)
        self.current_batch = 0
        self.pad_tokens = pad_tokens

    def __len__(self) -> int:
        return self.nbatch * self.batch_size

    def __getitem__(self, idx):

        idx = idx // self.batch_size
        start_idx = idx * self.bptt
        end_idx = (idx + 1) * self.bptt
        source = self.inputs[:, start_idx:end_idx, self.current_batch]

        # Targets already shifted
        target = self.target[:, start_idx:end_idx, self.current_batch]

        # We dont need it?
        # target = self._pad_target(target)

        self.current_batch += 1

        if self.current_batch == self.inputs.shape[2]:
            self.current_batch = 0

        return source, target

    def _pad_target(self, array: np.ndarray):
        """Pad n+1 ngram sequences at the end."""
        ngram = array.shape[0]

        if ngram == 1:
            return array

        for n_dim in range(2, ngram + 1):
            for shift in range(1, n_dim + 1):
                array[n_dim - 1][-shift] = self.pad_tokens[n_dim]

        return array


class GenericDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, bptt_size, pad_tokens, cpus=1):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.bptt_size = bptt_size
        self.cpus = cpus
        self.pad_tokens = pad_tokens

    def setup(self, stage):
        self.train = TextDataset(
            self.dataset["train"], self.batch_size, self.bptt_size, self.pad_tokens
        )
        self.test = TextDataset(
            self.dataset["test"], self.batch_size, self.bptt_size, self.pad_tokens
        )

        self.valid = TextDataset(
            self.dataset["validation"], self.batch_size, self.bptt_size, self.pad_tokens
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            # collate_fn=batch_collate,
            drop_last=False,
            num_workers=self.cpus,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            # collate_fn=batch_collate,
            drop_last=False,
            num_workers=self.cpus,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            # collate_fn=batch_collate,
            drop_last=False,
            num_workers=self.cpus,
            pin_memory=True,
        )


def get_text(x):
    return x["text"]


def filter_empty_row(example) -> bool:
    return len(example["source"]) > 0


def load_tokenized_dataset(
    ngme: str,
    ngram: int,
    model_type: str,
    max_dict_size: int,
    unk_threshold: int,
    fallback: bool,
    num_proc: int,
    is_forward: bool,
    packed: bool,
    *args,
    **kwargs,
) -> Tuple[dict, Dictionary]:
    """🤗"""

    # Check if we have a local config for local dataset
    if args[0] == "text" and args[1] in local_dataset_mapper:
        dataset = ld("text", data_files=local_dataset_mapper[args[1]])
    elif args[0].startswith("wikipedia"):
        dataset = ld(*local_dataset_mapper[args[0]]["args"])
    else:
        # Load the datasets from huggingface
        dataset = ld(*args, **kwargs)

    # TODO: What do we need from this?
    # dataset = preprocess(dataset)

    # TODO: Allow for pretrained dict
    print("Collecting dictionary...")
    with Timer(text=lambda secs: f"Elapsed time: {format_timespan(secs)}"):
        dictionary = load_dictionary_from_hf(
            ngme,
            dataset["train"],
            ngram,
            model_type,
            max_dict_size,
            unk_threshold,
            fallback,
            packed=packed,
        )

    print("Tokenize dataset...")
    tokenized_dataset = dataset.map(
        lambda x: dictionary.tokenize_line(x["text"], id_type=torch.int16, return_tensor="np"),
        load_from_cache_file=USE_CACHE,
        num_proc=num_proc,
    )

    print("Remove empty rows...")
    tokenized_dataset = tokenized_dataset.filter(filter_empty_row, num_proc=num_proc)
    tokenized_dataset.cleanup_cache_files()


    print("Concat rows to whole sequence")
    with ProgressBar():
        print("Train...")
        train_source = concat_dataset(tokenized_dataset["train"]["source"])
        train_target = concat_dataset(tokenized_dataset["train"]["target"])
        print("Test...")
        test_source = concat_dataset(tokenized_dataset["test"]["source"])
        test_target = concat_dataset(tokenized_dataset["test"]["target"])
        print("Valid...")
        valid_source = concat_dataset(tokenized_dataset["validation"]["source"])
        valid_target = concat_dataset(tokenized_dataset["validation"]["target"])

    dataset = {
        # "train": {"text": train, "source": train_source, "target": train_target},
        "train": {"source": train_source, "target": train_target},
        "test": {"source": test_source, "target": test_target},
        "validation": {"source": valid_source, "target": valid_target},
    }

    return dataset, dictionary


def calc_chunksize(iterable, num_workers):
    # obw_chunk_size = 1_000_000
    chunksize, extra = divmod(len(iterable), num_workers * 4)
    if extra:
        chunksize += 1
    return chunksize


def np_array(x):
    return np.array(x, dtype=np.int16)


def concat_dataset(rows: List[List[List[int]]]):
    # Numpy casts lists to float64, we therefore cannot safely donwscast to int16
    return np.concatenate(rows, axis=1, dtype=np.int16, casting="unsafe")


@jit(parallel=True)
def _concat_dataset(rows: List[List[List[int]]]):
    return np.concatenate(rows, axis=1, dtype=np.int16)


def load_dictionary_from_hf(
    ngme: str,
    source: List[str],
    ngrams: int,
    model_type: str,
    max_dict_size: int,
    unk_threshold: int,
    fallback: bool,
    packed: bool = False,
) -> Dictionary:

    dictionary = Dictionary(
        ngrams, max_dict_size, unk_threshold, fallback, ngme, packed=packed
    )

    if ngme == "sparse":
        populate_sparse_dict(dictionary, ngrams, model_type)
    elif ngme == "dense":
        populate_dense_dict(dictionary, ngrams, source)
    else:
        raise ValueError("NGME approach not known")

    if dictionary.max_dict_size > 0:
        dictionary = dictionary.unking()

    if ngme == "dense":
        for n_gram in range(2, dictionary.ngram + 1):
            start_idx = dictionary.add_ngram("<start>", n_gram)
            pad_idx = dictionary.add_ngram("<pad>", n_gram)
            dictionary.pad_tokens[n_gram] = pad_idx
            unk_idx = dictionary.add_ngram("<unk>", n_gram)
            dictionary.add_ngram(" " * n_gram, n_gram)

            if model_type == "transformer":
                # TODO: Needed?
                eod_idx = dictionary.add_ngram("<eod>", n_gram)

            if n_gram not in dictionary._marker_tokens:
                dictionary._marker_tokens[n_gram] = [start_idx, pad_idx, unk_idx]

    # Check if all unigrams were indexed first and all idx are consecutive
    assert list(dictionary.ngram2idx2word[1].keys()) == list(
        range(0, len(dictionary.ngram2idx2word[1]))
    )

    return dictionary


def populate_sparse_dict(dictionary, ngrams: int, model_type: str):
    """Build dictionary based on Akbik et. al character LM dict"""

    # Keep everything on the cpu
    flair_device = flair.device
    flair.device = "cpu"
    e = FlairEmbeddings("news-forward")

    dictionary.ngme = "sparse"

    for token in e.lm.dictionary.item2idx_not_encoded:
        for n_gram in range(1, ngrams + 1):
            dictionary.add_ngram(token, n_gram)

    for n_gram in range(1, ngrams + 1):
        dictionary.add_ngram("<start>", n_gram)
        dictionary.add_ngram("<pad>", n_gram)
        dictionary.add_ngram("<unk>", n_gram)

        if model_type == "transformer":
            _ = dictionary.add_ngram("<eod>", n_gram)

    del e
    flair.device = flair_device


def collect_ngrams(line, n, dictionary):

    ngrams = []

    for n_gram in nltk.ngrams(line, n):
        for c in n_gram:
            if not c in dictionary.ngram2word2idx[1]:
                break
        else:
            ngrams.append("".join(n_gram))

    return ngrams


def populate_dense_dict(dictionary: Dictionary, ngrams: int, source: List[str]):

    dictionary.ngme = "dense"

    flair_device = flair.device
    flair.device = "cpu"
    # Using unigrams from flair as base
    e = FlairEmbeddings("news-forward")

    # Guarantee that all unigram tokens are indexed first
    # Uni-gram tokens
    for token in e.lm.dictionary.item2idx_not_encoded:
        dictionary.add_ngram(token, 1)

    start_idx = dictionary.add_ngram("<start>", 1)
    pad_idx = dictionary.add_ngram("<pad>", 1)
    unk_idx = dictionary.add_ngram("<unk>", 1)
    dictionary.add_ngram(" ", 1)

    if 1 not in dictionary._marker_tokens:
        dictionary._marker_tokens[1] = [start_idx, pad_idx, unk_idx]

    # Add new n-gram token only if all uni-gram parts exist
    for n in range(1, ngrams + 1):

        if n == 1:
            for line in source:
                for c in line:
                    if c in dictionary.ngram2word2idx[1]:
                        dictionary.frequencies.update({c: 1})
        else:
            for line in source:
                ngram_lists = collect_ngrams(line, n, dictionary)
                for ngram in ngram_lists:
                    dictionary.add_ngram(ngram, n)

    del e
    flair.device = flair_device


# def preprocess(dataset):
#     print("Preprocess dataset...")
#     with Timer(text=lambda secs: f"Elapsed time: {format_timespan(secs)}"):
#
#     # TODO: Refactor
#     if args[0].startswith("wikipedia"):
#         train = []
#         test = []
#         valid = []
#
#         wiki206 = 120000  # For english about 2x wiki103
#
#         for row in tqdm(dataset["train"]["text"][:wiki206]):
#             train.append(row)
#
#         for row in tqdm(dataset["train"]["text"][wiki206 + 1 : wiki206 + 1000]):
#             test.append(row)
#
#         for row in tqdm(dataset["train"]["text"][wiki206 + 1001 : wiki206 + 2000]):
#             valid.append(row)
#     else:
#         train = []
#         for row in tqdm(dataset["train"]):
#             train.append(row["text"])
#
#         if "test" in dataset:
#             test = process_map(get_text, dataset["test"], max_workers=num_proc)
#         else:
#             test = []
#
#         if "validation" in dataset:
#             valid = process_map(
#                 get_text, dataset["validation"], max_workers=num_proc
#             )
#         else:
#             valid = []
#     dataset.set_format(type="numpy")
#     print(dataset)
#     exit()
#     sample = 1
#
#     if sample != 1.0:
#         print(f"Subsample to {sample}...")
#         with Timer(text=lambda secs: f"Elapsed time: {format_timespan(secs)}"):
#             train = train[0 : int(len(train) * sample)]
#
#     if not is_forward:
#         print("Revert text sequence...")
#         with Timer(text=lambda secs: f"Elapsed time: {format_timespan(secs)}"):
#                 train = train[::-1]
#                 test = test[::-1]
#                 valid = valid[::-1]
#
#     # We basically use HF datasets here used for the multiprocessing feature for the tokenization
#     dataset = DatasetDict(
#         {
#             "train": HfDataset.from_dict({"text": train}),
#             "test": HfDataset.from_dict({"text": test}),
#             "validation": HfDataset.from_dict({"text": valid}),
#         }
#     )
#     pass
