import gc
import multiprocessing
import pathlib
from collections import Iterable
from functools import partial
from typing import List, Optional, Tuple

import dask.array as da
import flair
import nltk
import numpy as np
import pytorch_lightning as pl
import torch
import webdataset as wds
from codetiming import Timer
from datasets import load_dataset as ld
from datasets.dataset_dict import DatasetDict
from flair.embeddings.token import FlairEmbeddings
from humanfriendly import format_timespan
from numba import jit
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset, Dataset
from tqdm import tqdm

from . import USE_CACHE
from .data import batchify, local_dataset_mapper
from .dictionary import Dictionary
from .utils import split_range


class TextDataset(Dataset):
    def __init__(self, ds, batch_size, bptt_size, pad_tokens) -> None:
        self.bptt = bptt_size
        self.batch_size = batch_size
        self.inputs, self.nbatch = batchify(ds["source"], batch_size, bptt_size)
        self.target, _ = batchify(ds["target"], batch_size, bptt_size)

        assert isinstance(self.inputs, torch.Tensor)
        assert isinstance(self.target, torch.Tensor)

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

            # TODO: Is this to often?
            gc.collect()

        assert isinstance(source, torch.Tensor)
        assert isinstance(target, torch.Tensor)

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
        self.train = ConcatDataset(
            [
                TextDataset(data, self.batch_size, self.bptt_size, self.pad_tokens)
                for data in self.dataset["train"]
            ]
        )
        self.test = ConcatDataset(
            [
                TextDataset(data, self.batch_size, self.bptt_size, self.pad_tokens)
                for data in self.dataset["test"]
            ]
        )
        self.valid = ConcatDataset(
            [
                TextDataset(data, self.batch_size, self.bptt_size, self.pad_tokens)
                for data in self.dataset["validation"]
            ]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.cpus,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.cpus,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.cpus,
            pin_memory=True,
        )


def get_text(x):
    return x["text"]


def filter_empty_row(example) -> bool:
    return len(example["text"]) > 0


def load_dataset_from_source(ds_path: str) -> DatasetDict:
    """We using the HF dataset path convenientself.
    Usually:
    <dataset>/<subset>

    For loading local datset, use:
    text/<dataset-path>

    We map <dataset-path> to dict of target files.
    """

    prefix, subset = ds_path.split("/")

    print(prefix, subset)

    # Check if we have a local config for local dataset
    if prefix == "text" and subset in local_dataset_mapper:
        dataset = ld("text", data_files=local_dataset_mapper[subset])
    elif prefix.startswith("wikipedia"):
        dataset = ld(*local_dataset_mapper[prefix]["args"])
    else:
        # Load the datasets from huggingface
        dataset = ld(*ds_path.split("/"))

    assert isinstance(dataset, DatasetDict)

    return dataset


def process_tokenized_dataset(
    dataset_path: str,
    ngme: str,
    ngram: int,
    model_type: str,
    max_dict_size: int,
    num_proc: int,
    is_forward: bool,
    packed: bool,
    dict_file_name: Optional[str] = None,
    rows_threshold: int = 50_000_000,
    **kwargs,
) -> Iterable[Tuple[dict, Dictionary]]:
    """🤗"""

    dataset = load_dataset_from_source(dataset_path)

    # TODO: What do we need from this?
    # dataset = preprocess(dataset)

    print("Collecting dictionary...")
    if dict_file_name:
        print(f"Reusing dict: {dict_file_name}")
        dictionary = load_dictionary_from_file(dict_file_name)
    else:
        with Timer(text=lambda secs: f"Elapsed time: {format_timespan(secs)}"):
            dictionary = load_dictionary_from_hf(
                ngme,
                dataset["train"]["text"],
                ngram,
                model_type,
                max_dict_size,
                packed=packed,
                num_workers=num_proc,
            )

    def tokenize(x):
        result = dictionary.tokenize_line(
            x["text"], id_type=torch.int16, return_tensor="np"
        )
        return result

    print("Remove empty rows...")
    dataset = dataset.filter(filter_empty_row, num_proc=num_proc)

    print("Tokenize dataset...")
    tokenized_dataset = dataset.map(
        tokenize,
        load_from_cache_file=USE_CACHE,
        num_proc=num_proc,
    )

    # Until this point big datasets can all be processed efficiently
    # But the concatenation is hard
    # Check here for length and split and yield datasetsubsets
    # Train is the largest and mostly only problem

    train_len = len(tokenized_dataset["train"])

    if train_len > rows_threshold:
        extra_split = (train_len / rows_threshold) != (train_len // rows_threshold)
        num_splits = train_len // rows_threshold

        num_splits = train_len // rows_threshold

        if extra_split:
            num_splits += 1

        for i in range(num_splits + 1):
            print(f"Processing split {i}/{num_splits}")
            train_start, train_end = split_range(
                i, tokenized_dataset["train"], num_splits
            )
            test_start, test_end = split_range(i, tokenized_dataset["test"], num_splits)
            valid_start, valid_end = split_range(
                i, tokenized_dataset["validation"], num_splits
            )

            print(
                f"Taking splits:\n    \
                    Train: [{train_start}:{train_end}]\n    \
                    Test:  [{test_start}/{test_end}]\n    \
                    Valid: [{valid_start}/{valid_end}]"
            )

            if (
                (test_end - test_start) <= 1
                or (train_end - train_start) <= 1
                or (valid_end - valid_start) <= 1
            ):
                print("Skipping last spit")
                break

            print("Train...")
            train_source, train_target = concat_from_split(
                tokenized_dataset["train"][train_start:train_end]
            )
            print("Test...")
            test_source, test_target = concat_from_split(
                tokenized_dataset["test"][test_start:test_end]
            )
            print("Valid...")
            valid_source, valid_target = concat_from_split(
                tokenized_dataset["validation"][valid_start:valid_end]
            )

            # To avoid too small batches of bptt, we dont return too small datasets
            # Common batch: [100, 150] = 15_000 chars for one batch
            # Average obw news row has
            print(f"train len: {train_source.shape}")
            dataset = {
                "train": {"source": train_source, "target": train_target},
                "test": {"source": test_source, "target": test_target},
                "validation": {"source": valid_source, "target": valid_target},
            }

            yield dataset, dictionary
    else:
        print("Train...")
        train_source, train_target = concat_from_split(tokenized_dataset["train"])
        print("Test...")
        test_source, test_target = concat_from_split(tokenized_dataset["test"])
        print("Valid...")
        valid_source, valid_target = concat_from_split(tokenized_dataset["validation"])

        dataset = {
            "train": {"source": train_source, "target": train_target},
            "test": {"source": test_source, "target": test_target},
            "validation": {"source": valid_source, "target": valid_target},
        }
        yield dataset, dictionary


def write_tokenized_dataset(split_iterator, path):
    shard_path = f"{path}-%0d.tar"
    with wds.ShardWriter(shard_path) as sink:
        for i, (ds, dictionary) in enumerate(split_iterator):
            for train_split in ["train", "test", "validation"]:
                if train_split not in ds:
                    print(f"Split {train_split} not found. Skipping.")

                ds_split = ds[train_split]
                source = ds_split["source"]
                target = ds_split["target"]

                sink.write(
                    {
                        "__key__": f"{path}_{train_split}_{str(i)}",
                        "source.pyd": source,
                        "target.pyd": target,
                        "split": train_split,
                    }
                )
    return dictionary, shard_path


def load_tokenized_dataset(path: str) -> dict:
    ds_dir = pathlib.Path(path)

    assert ds_dir.is_file(), "Tokenized dataset has to be a tar archive"

    dataset = wds.WebDataset(path).decode("rgb8")

    dict_dataset = {"train": [], "test": [], "validation": []}

    # Collect splitted dataset
    for split in dataset:
        dict_dataset[split["split"].decode("utf-8")].append(
            {
                "source": split["source.pyd"],
                "target": split["target.pyd"],
            }
        )

    return dict_dataset


def calc_chunksize(iterable, num_workers):
    # obw_chunk_size = 1_000_000
    chunksize, extra = divmod(len(iterable), num_workers * 4)
    if extra:
        chunksize += 1
    return chunksize


def np_array(x):
    return np.array(x, dtype=np.int16)


def concat_from_split(split):
    source = split["source"]
    target = split["target"]

    source_array = concat_dataset(source)
    target_array = concat_dataset(target)

    return source_array, target_array


def concat_dataset(lst):
    sublists = []

    for sublist in tqdm(lst):
        sublists.append(da.from_array(sublist, chunks=(len(sublist[0]), len(sublist))))

    return da.concatenate(sublists, axis=1).compute()


def concat_dataset(rows: List[List[List[int]]]):
    # Numpy casts lists to float64, we therefore cannot safely donwscast to int16
    return np.concatenate(rows, axis=1, dtype=np.int16, casting="unsafe")


@jit(parallel=True)
def _concat_dataset_(rows: List[List[List[int]]]):
    return np.concatenate(rows, axis=1, dtype=np.int16)


def load_dictionary_from_file(dict_file_name: str):
    return Dictionary.load_from_file(dict_file_name)


def load_dictionary_from_hf(
    ngme: str,
    source: List[str],
    ngrams: int,
    model_type: str,
    max_dict_size: int,
    packed: bool = False,
    num_workers: int = 1,
) -> Dictionary:
    dictionary = Dictionary(ngrams, max_dict_size, ngme)

    if ngme == "sparse":
        populate_sparse_dict(dictionary, ngrams)
    elif ngme == "dense":
        populate_dense_dict(dictionary, ngrams, source, num_workers)
    else:
        raise ValueError("NGME approach not known")

    if dictionary.max_dict_size == 0:
        dictionary.max_dict_size = len(dictionary)

    if ngme == "dense":
        print("Apply unking...")
        dictionary = dictionary.unking()

    # Check if all unigrams were indexed first and all idx are consecutive
    assert list(dictionary.ngram2idx2word[1].keys()) == list(
        range(0, len(dictionary.ngram2idx2word[1]))
    )

    return dictionary


def populate_sparse_dict(dictionary, ngrams: int):
    """Build dictionary based on Flair character LM dict"""

    unigram_tokens = get_unigram_tokens()
    for n_gram in range(1, ngrams + 1):
        dictionary.add_ngram("<start>", n_gram)
        dictionary.add_ngram("<pad>", n_gram)
        dictionary.add_ngram("<unk>", n_gram)

        for token in unigram_tokens:
            dictionary.add_ngram(token, n_gram)

    # for n_gram in range(1, ngrams + 1):
    #     if model_type == "transformer":
    #         _ = dictionary.add_ngram("<eod>", n_gram)
    #


def collect_ngrams(line, n):
    return ["".join(ngram) for ngram in nltk.ngrams(line, n)]

    # for n_gram in :
    #     for c in n_gram:
    #         if not c in dictionary.ngram2word2idx[1]:
    #             break
    #     else:
    #         ngrams.append("".join(n_gram))
    #
    # return ngrams
    #


def populate_dense_dict(
    dictionary: Dictionary, ngrams: int, source: List[str], num_workers: int = 1
):
    dictionary.ngme = "dense"

    # Guarantee that all unigram tokens are indexed first
    # Uni-gram tokens
    for token in get_unigram_tokens():
        dictionary.add_ngram(token, 1)

    # Add new n-gram token only if all uni-gram parts exist
    for n in range(1, ngrams + 1):
        start_idx = dictionary.add_ngram("<start>", n)
        pad_idx = dictionary.add_ngram("<pad>", n)
        unk_idx = dictionary.add_ngram("<unk>", n)
        dictionary.add_ngram(" " * n, n)
        dictionary._marker_tokens[n] = [start_idx, pad_idx, unk_idx]

    ngram_list = list(range(1, ngrams + 1))
    if num_workers > 1:
        print(f"Run dictionary collection with {num_workers} workers")
        with multiprocessing.Pool(num_workers) as pool:
            for ngram_tokens in tqdm(
                pool.map(
                    partial(add_ngrams_from_text, ngrams=ngram_list),
                    source,
                    chunksize=25,
                )
            ):
                for n, tokens in ngram_tokens.items():
                    dictionary.add_ngrams(tokens, n)

    else:
        for line in source:
            add_ngrams_from_text(line, ngram_list)

    return dictionary


def add_ngrams_from_text(text: str, ngrams: List[int]):
    return {ngram: collect_ngrams(text, ngram) for ngram in ngrams}


def get_unigram_tokens() -> List[str]:
    flair_device = flair.device
    flair.device = "cpu"

    # Using unigrams from flair as base
    e = FlairEmbeddings("news-forward")
    flair.device = flair_device
    return list(e.lm.dictionary.item2idx_not_encoded.keys())


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
