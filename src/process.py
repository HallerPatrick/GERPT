import collections
from functools import partial
import os
import multiprocessing
import pathlib
from typing import List, Optional, Tuple
from collections.abc import Iterable

from codetiming import Timer
from datasets.dataset_dict import DatasetDict
import flair
from flair.embeddings import FlairEmbeddings
from humanfriendly import format_timespan
import nltk

import h5py
import numpy as np
import torch
from tqdm import tqdm
import webdataset as wds
from webdataset.tariterators import braceexpand

from src.utils import split_range, set_stacked_numpy_formatter

from . import USE_CACHE
from src.dataset import load_dataset_from_source
from src.dictionary import Dictionary


def process_tokenized_dataset(
        target_path,
        dataset_path: str,
        ngme: str,
        ngram: int,
        model_type: str,
        max_dict_size: int,
        num_proc: int,
        is_forward: bool,
        packed: bool,
        dict_file_name: Optional[str] = None,
        write_strategy: str = "memmap",
        rows_threshold: int = 50_000_000,
        **kwargs,
) -> Tuple[Dictionary, Optional[Iterable]]:
    """🤗"""

    return _process_tokenized_dataset(
        target_path,
        dataset_path,
        ngme,
        ngram,
        model_type,
        max_dict_size,
        num_proc,
        is_forward,
        packed,
        dict_file_name,
        write_strategy,
        rows_threshold,
        **kwargs
    )


def _process_tokenized_dataset(
        target_path,
        dataset_path: str,
        ngme: str,
        ngram: int,
        model_type: str,
        max_dict_size: int,
        num_proc: int,
        is_forward: bool,
        packed: bool,
        dict_file_name: Optional[str] = None,
        write_strategy: str = "memmap",
        rows_threshold: int = 50_000_000,
        **kwargs,
):
    dataset = load_dataset_from_source(dataset_path)

    if isinstance(dataset, collections.Iterable):
        assert dict_file_name, "Can only preprocess splits with predefined dictionary"

        dictionary = load_dictionary_from_file(dict_file_name)

        if not os.path.exists(target_path):
            os.mkdir(target_path)

        for i, dataset_split in enumerate(dataset):
            write_file_split(dataset_split, dictionary, target_path, i, num_proc)
        return None, None

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
        return {**result, "text_len": len(x["text"])}

    print("Remove empty rows...")
    dataset = dataset.filter(filter_empty_row, num_proc=num_proc)

    print("Tokenize dataset...")
    tokenized_dataset = dataset.map(
        tokenize,
        load_from_cache_file=USE_CACHE,
        num_proc=num_proc,
    )

    # TODO; Does this help us in anyway?

    if write_strategy == "sharding":
        return dictionary, process_with_shards(tokenized_dataset, rows_threshold)

    if write_strategy == "memmap":
        new_write_tokenized_dataset(tokenized_dataset, target_path)

    if write_strategy == "hdf5":
        write_to_hdf5(tokenized_dataset, target_path)

    return dictionary, None


def write_file_split(dataset, dictionary, target_path, idx, num_proc):
    print("Remove empty rows...")
    dataset = dataset.filter(filter_empty_row, num_proc=num_proc)

    def tokenize(x):
        result = dictionary.tokenize_line(
            x["text"], id_type=torch.int16, return_tensor="np"
        )
        return {**result, "text_len": len(x["text"])}

    print("Tokenize dataset...")
    tokenized_dataset = dataset.map(
        tokenize,
        load_from_cache_file=USE_CACHE,
        num_proc=num_proc,
    )

    print("Concat dataset...")
    source, target = concat_from_split(tokenized_dataset["train"])

    target_file = pathlib.Path(target_path) / f"train-{str(idx)}"
    print(f"Saving split {str(idx)} to {target_file}")
    torch.save({
        "source": source,
        "target": target
    }, target_file)

def load_from_splits(path) -> collections.Iterator:
    path = pathlib.Path(path)

    idxs = [int(file.stem.removeprefix("train-")) for file in path.iterdir() if file.is_file()]
    idxs.sort()

    for idx in idxs:
        ds_dict = torch.load(path / f"train-{str(idx)}")
        empty_split = {"source": np.array([]), "target": np.array([])}
        yield {
            "train": ds_dict,
            "test": empty_split, 
            "validation": empty_split
        }


def filter_empty_row(example) -> bool:
    return len(example["text"]) > 0


def process_with_shards(tokenized_dataset: DatasetDict, rows_threshold: int):
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
                    Test:  [{test_start}:{test_end}]\n    \
                    Valid: [{valid_start}:{valid_end}]"
            )

            if (train_end - train_start) <= 1:
                print("Skipping last spit")
                break

            if (test_end - test_start) <= 1:
                test_source, test_target = np.array([]), np.array([])

            if (valid_end - valid_start) <= 1:
                valid_source, valid_target = np.array([]), np.array([])

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

            yield dataset


def write_sharded_tokenized_dataset(split_iterator, path):
    shard_path = f"{path}-%0d.tar"
    with wds.ShardWriter(shard_path) as sink:
        for i, ds in enumerate(split_iterator):
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
    return shard_path


def load_sharded_splits(path):
    def collate_fn(lst):
        sample = {}
        for split in lst:
            split_name = split[2].decode("utf-8")
            sample[split_name] = {
                "source": split[0],
                "target": split[1],
                "split": split_name
            }
        return sample

    # For some reason we have to check ourself how many shards we have
    # Otherwise we iter through all shards for ever
    urls = list(braceexpand.braceexpand(path))

    dataset = wds.DataPipeline(
        wds.ResampledShards(urls, nshards=len(urls)),
        wds.tarfile_to_samples(),
        wds.decode("rgb"),
        wds.to_tuple("source.pyd", "target.pyd", "split"),
        wds.batched(3, collation_fn=collate_fn),
    )

    return dataset


def load_sharded_tokenized_dataset(path: str) -> dict:
    ds_dir = pathlib.Path(path)

    # assert ds_dir.is_file(), "Tokenized dataset has to be a tar archive"

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


def len_row(row):
    return len(row[0])


def calculate_total_seq_length(dataset_split):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as pool:
        results = pool.map(len_row, dataset_split["source"])
    return sum(results)


def write_to_hdf5(tokenized_dataset, path):
    # tokenized_dataset = tokenized_dataset.remove_columns("text")
    # set_stacked_numpy_formatter()
    # tokenized_dataset.set_format("snp")

    f = h5py.File(path + ".h5", "w")

    for split in ["train", "test", "validation"]:

        ngram = len(tokenized_dataset[split]["source"][0])
        total_train_len = sum(tokenized_dataset[split]["text_len"])

        for seq_name in ["source", "target"]:

            dataset = f.create_dataset(split.replace("validation", "valid") + "_" + seq_name, (ngram, total_train_len))

            offset = 0
            # print(split, seq_name)
            for lst in tqdm(tokenized_dataset[split][seq_name]):
                array = np.array(lst)
                array_len = array.shape[1]
                dataset[:, offset: (offset + array_len)] = array[:]
                offset += array_len
                del array


def load_hdf5(path):
    dataset = {}
    hd = h5py.File(path, "r")

    for gname, group in hd.items():
        split, seq_name = gname.split("_")

        if split not in dataset:
            dataset[split] = {}
        dataset[split][seq_name] = group[:]

    return dataset


def new_write_tokenized_dataset(dataset, path):
    os.mkdir(path)

    total_train_len = 0

    for seq_name in ["source", "target"]:

        ngram = len(dataset["train"][seq_name][0])

        if total_train_len == 0:
            print("Calculate seq length...")
            total_train_len = sum(dataset["train"]["text_len"])
            # total_train_len = calculate_total_seq_length(dataset["train"])
            print(f"Total size: {(ngram, total_train_len)}")

        fp_train_source = np.memmap(f"{path}/train_{seq_name}.npy", dtype='int16', mode='w+',
                                    shape=(ngram, total_train_len))

        offset = 0
        print("Write rows to file...")
        for array in tqdm(dataset["train"][seq_name]):
            array_len = array.shape[1]
            fp_train_source[:, offset: (offset + array_len)] = array[:]
            offset += array_len

        fp_train_source.flush()

    with open(f"{path}/size.txt", "w") as f:
        f.write(f"{ngram},{total_train_len}")

    source, target = concat_from_split(dataset["test"])
    np.save(f"{path}/test_source", source)
    np.save(f"{path}/test_target", target)

    source, target = concat_from_split(dataset["validation"])
    np.save(f"{path}/valid_source", source)
    np.save(f"{path}/valid_target", target)


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


def load_tokenized_dataset(path: str, ngram: int) -> dict:
    dataset = {
        "train": {},
        "test": {},
        "valid": {}
    }

    ngram, train_size = open(path + "/size.txt", "r").read().strip().split(",")
    ngram = int(ngram)
    train_size = int(train_size)

    for split in ["train", "test", "valid"]:
        for seq_name in ["source", "target"]:
            if split == "train":
                array = np.memmap(f"{path}/{split}_{seq_name}.npy", shape=(ngram, train_size))
                array = array.reshape((3, -1))
            else:
                array = np.load(f"{path}/{split}_{seq_name}.npy", allow_pickle=True)
            dataset[split][seq_name] = array

    return dataset


def _load_tokenized_dataset(path: str) -> dict:
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


def concat_dataset(rows: List[List[List[int]]]):
    # Numpy casts lists to float64, we therefore cannot safely donwscast to int16
    return np.concatenate(rows, axis=1, dtype=np.int16, casting="unsafe")


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
            for n, tokens in add_ngrams_from_text(line, ngram_list).items():
                dictionary.add_ngrams(tokens, n)

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
