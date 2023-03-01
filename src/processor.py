"""Processor for the dataset"""
import os

from typing import Dict, Union, Iterable, List
from pathlib import Path

from functools import partial

import h5py
import braceexpand
import torch
import numpy as np
from tqdm import tqdm
from datasets import DatasetDict
import webdataset as wds

from src.dictionary import Dictionary
from src.utils import split_range


class InvalidPathError(Exception):
    """Raised when the path is invalid for specific processor""" ""

def filter_empty_row(example):
    return len(example["text"]) > 0

def tokenize(row, dictionary: Dictionary):
    result = dictionary.tokenize_line(
        row["text"], id_type=torch.int16, return_tensor="np"
    )
    return {**result, "text_len": len(row["text"])}

class Processor:
    def __init__(
        self,
        target_path: str,
        dictionary: Dictionary,
        ngram: int,
        is_forward: bool,
        num_procs: int,
        **kwargs,
    ):
        self.target_path = target_path
        self.dictionary = dictionary
        self.ngram = ngram
        self.is_forward = is_forward
        self.kwargs = kwargs
        self.num_procs = num_procs

    def run(self, dataset):
        self._mkdir(path=self.target_path)
        dataset = self.process_data(dataset)
        self.write_dataset(dataset)

    @staticmethod
    def read_dataset(file_path: str) -> Union[DatasetDict, Iterable[DatasetDict]]:
        """Reads the preprocessed dataset from the file path"""

        strategy = Processor.get_strategy_from_path(file_path)

        return Processor.from_strategy(strategy).read_dataset(file_path)

    @staticmethod
    def get_strategy_from_path(file_path: str) -> str:
        """Reads the strategy from the file path"""

        with open(file_path + "/strategy", "r", encoding="utf8") as f:
            strategy = f.read().strip()

        return strategy

    def write_dataset(self, dataset: DatasetDict):
        """Writes the dataset to the file path"""

    def process_data(self, dataset: DatasetDict) -> DatasetDict:
        """
        Processing:
        1. Preprocess the data
        2. Tokenize the data
        """

        dataset = dataset.filter(filter_empty_row, num_proc=self.num_procs)
        dataset = dataset.map(partial(tokenize, dictionary=self.dictionary), num_proc=self.num_procs)

        return dataset

    @staticmethod
    def validate_path(file_path: str) -> bool:
        """Validates the file path"""

    @staticmethod
    def from_strategy(strategy: str) -> "Processor":
        """Factory method for creating a processor from a strategy"""

        if strategy == "default":
            return DefaultProcessor
        # elif strategy == "memmap":
        #     return MmapProcessor
        elif strategy == "webdataset":
            return ShardProcessor
        elif strategy == "hdf5":
            return HDF5Processor
        elif strategy == "split":
            return SplitProcessor
        else:
            raise NotImplementedError(f"Strategy {strategy} not implemented")

    def _mkdir(self, path: str):
        """Creates the directory if it does not exist"""
        if not os.path.exists(path):
            os.makedirs(path)

def concat_from_split(split):
    source = split["source"]
    target = split["target"]

    source_array = concat_dataset(source)
    target_array = concat_dataset(target)

    return source_array, target_array


def concat_dataset(rows: List[List[List[int]]]):
    # Numpy casts lists to float64, we therefore cannot safely donwscast to int16
    return np.concatenate(rows, axis=1, dtype=np.int16, casting="unsafe")


class DefaultProcessor(Processor):
    """Default processor for datasets tries to simply concatenate the text columns
    with np.concatenate and saves with torch.save""" ""

    def __init__(
        self,
        target_path: str,
        dictionary: Dictionary,
        ngram: int,
        is_forward: bool,
        num_procs: int,
        **kwargs,
    ):
        super().__init__(
            target_path, dictionary, ngram, is_forward, num_procs, **kwargs
        )

    @staticmethod
    def read_dataset(file_path: str) -> DatasetDict:
        """Reads the dataset from the file path or huggingface"""

        file_dir = Path(file_path)

        dict_dataset = {}

        for file in file_dir.glob("*.pt"):
            print(f"Reading {file}...", end="")
            data = torch.load(file)
            print("Done")
            dict_dataset[file.stem] = data

        return DatasetDict(dict_dataset)

    def write_dataset(self, dataset: DatasetDict()):
        """Writes the dataset to the file path"""

        
        with open(f"{self.target_path}/strategy", "w") as f:
            f.write("default")

        for split in dataset:
            target_file = Path(self.target_path) / f"{split}.pt"
            print(f"Saving split to {str(target_file)}...", end="")
            source, target = concat_from_split(dataset[split])

            torch.save({"source": source, "target": target}, target_file)

            print("Done")

    @staticmethod
    def validate_path(file_path: str) -> bool:
        """Validates the file path"""

        path = Path(file_path)

        if not path.is_dir():
            return False

        # Path should contain a train.txt, valid.txt and test.txt
        if not (path / "train.txt").is_file():
            return False

        if not (path / "valid.txt").is_file():
            return False

        if not (path / "test.txt").is_file():
            return False

        return True


class MmapProcessor(Processor):


    @staticmethod
    def read_dataset(file_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        dataset = {"train": {}, "test": {}, "valid": {}}

        ngram, train_size = open(file_path + "/size.txt", "r").read().strip().split(",")
        ngram = int(ngram)
        train_size = int(train_size)

        for split in ["train", "test", "valid"]:
            for seq_name in ["source", "target"]:
                if split == "train":
                    array = np.memmap(
                        f"{file_path}/{split}_{seq_name}.npy", shape=(ngram, train_size)
                    )
                    array = array.reshape((ngram, -1))
                else:
                    array = np.load(
                        f"{file_path}/{split}_{seq_name}.npy", allow_pickle=True
                    )
                dataset[split][seq_name] = array

        return dataset

    def write_dataset(self, dataset):
        try:
            os.mkdir(self.target_path)
        except FileExistsError:
            pass

        total_train_len = 0

        for seq_name in ["source", "target"]:
            ngram = len(dataset["train"][seq_name][0])

            if total_train_len == 0:
                print("Calculate seq length...")
                total_train_len = sum(dataset["train"]["text_len"])
                # total_train_len = calculate_total_seq_length(dataset["train"])
                print(f"Total size: {(ngram, total_train_len)}")

            fp_train_source = np.memmap(
                f"{self.target_path}/train_{seq_name}.npy",
                dtype="int16",
                mode="w+",
                shape=(ngram, total_train_len),
            )

            offset = 0
            print("Write rows to file...")
            for array in tqdm(dataset["train"][seq_name]):
                array = np.array(array, dtype=np.int16)
                array_len = array.shape[1]
                fp_train_source[:, offset : (offset + array_len)] = array[:]
                offset += array_len

            fp_train_source.flush()

        with open(f"{self.target_path}/size.txt", "w") as f:
            f.write(f"{ngram},{total_train_len}")

        source, target = concat_from_split(dataset["test"])
        np.save(f"{self.target_path}/test_source", source)
        np.save(f"{self.target_path}/test_target", target)

        source, target = concat_from_split(dataset["valid"])
        np.save(f"{self.target_path}/valid_source", source)
        np.save(f"{self.target_path}/valid_target", target)


class HDF5Processor(Processor):

    def write_dataset(self, dataset):
        f = h5py.File(self.target_path + ".h5", "w")

        for split in ["train", "test", "validation"]:

            ngram = len(dataset[split]["source"][0])
            total_train_len = sum(dataset[split]["text_len"])

            for seq_name in ["source", "target"]:

                h5_dataset = f.create_dataset(split.replace("validation", "valid") + "_" + seq_name, (ngram, total_train_len))

                offset = 0
                # print(split, seq_name)
                for lst in tqdm(dataset[split][seq_name]):
                    array = np.array(lst)
                    array_len = array.shape[1]
                    h5_dataset[:, offset: (offset + array_len)] = array[:]
                    offset += array_len
                    del array

    @staticmethod
    def read_dataset(file_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        dataset = {"train": {}, "test": {}, "valid": {}}

        hd = h5py.File(file_path + ".h5", "r")

        for gname, group in hd.items():
            split, seq_name = gname.split("_")

            if split not in dataset:
                dataset[split] = {}
            dataset[split][seq_name] = group[:]

        return dataset


class ShardProcessor(Processor):

    def __init__(self, target_path, dictionary, ngram, is_forward, num_procs, **kwargs):
        super().__init__(target_path, dictionary, ngram, is_forward, num_procs, **kwargs)

        if "row_threshold" in kwargs:
            self.rows_threshold = kwargs["rows_threshold"]

    @staticmethod
    def read_dataset(file_path: str):
        """Loads the dataset from the file path"""

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
        urls = list(braceexpand.braceexpand(file_path))

        dataset = wds.DataPipeline(
            wds.ResampledShards(urls, nshards=len(urls)),
            wds.tarfile_to_samples(),
            wds.decode("rgb"),
            wds.to_tuple("source.pyd", "target.pyd", "split"),
            wds.batched(3, collation_fn=collate_fn),
        )

        return dataset

    def write_dataset(self, dataset: Union[DatasetDict, Iterable[DatasetDict]]):
        
        if isinstance(dataset, DatasetDict):
            dataset_iterator = self._split_dataset(dataset)
        
        shard_path = f"{self.target_path}-%0d.tar"

        with wds.ShardWriter(shard_path) as sink:
            for i, ds in enumerate(dataset_iterator):
                for train_split in ["train", "test", "valid"]:
                    if train_split not in ds:
                        print(f"Split {train_split} not found. Skipping.")

                    ds_split = ds[train_split]
                    source = ds_split["source"]
                    target = ds_split["target"]

                    sink.write(
                        {
                            "__key__": f"{self.target_path}_{train_split}_{str(i)}",
                            "source.pyd": source,
                            "target.pyd": target,
                            "split": train_split,
                        }
                    )
        return shard_path

    def _split_dataset(self, dataset: DatasetDict) -> List[DatasetDict]:
        """Splits the dataset into shards"""

        train_len = len(dataset["train"])

        if not hasattr(self, "rows_threshold"):
            self.rows_threshold = train_len

        extra_split = (train_len / self.rows_threshold) != (train_len // self.rows_threshold)
        num_splits = train_len // self.rows_threshold

        num_splits = train_len // self.rows_threshold

        if extra_split:
            num_splits += 1

        for i in range(num_splits + 1):
            print(f"Processing split {i}/{num_splits}")
            train_start, train_end = split_range(
                i, dataset["train"], num_splits
            )
            test_start, test_end = split_range(i, dataset["test"], num_splits)
            valid_start, valid_end = split_range(
                i, dataset["valid"], num_splits
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
                dataset["train"][train_start:train_end]
            )
            print("Test...")
            test_source, test_target = concat_from_split(
                dataset["test"][test_start:test_end]
            )
            print("Valid...")
            valid_source, valid_target = concat_from_split(
                dataset["validation"][valid_start:valid_end]
            )

            dataset = {
                "train": {"source": train_source, "target": train_target},
                "test": {"source": test_source, "target": test_target},
                "valid": {"source": valid_source, "target": valid_target},
            }

            yield dataset


class SplitProcessor(Processor):

    @staticmethod
    def read_dataset(file_path: str):
        path = Path(file_path)

        idxs = [int(file.stem.removeprefix("train-")) for file in path.iterdir() if file.is_file() and file.stem != "strategy"]
        idxs.sort()

        for idx in idxs:
            ds_dict = torch.load(path / f"train-{str(idx)}")
            empty_split = {"source": np.array([]), "target": np.array([])}
            yield {
                "train": ds_dict,
                "test": empty_split, 
                "valid": empty_split
            }

    def run(self, dataset: Iterable[DatasetDict]):
        self._mkdir(path=self.target_path)

        with open(f"{self.target_path}/strategy", "w") as f:
            f.write("split")

        for idx, ds in enumerate(dataset):
            print(f"Processing split {idx}")
            print("-"*80)
            ds = self.process_data(ds)
            self._write_dataset(ds, idx)
            print("-"*80)
            
    def _write_dataset(self, dataset: Iterable[DatasetDict], idx: int):

        for train_split in ["train", "test", "valid"]:
            if train_split not in dataset:
                print(f"Split {train_split} not found. Skipping.")
                continue
            source, target = concat_from_split(dataset[train_split])
            self._write_split(source, target, idx, train_split)

    def _write_split(self, source, target, idx: int, split: str):
        target_file = Path(self.target_path) / f"{split}-{str(idx)}"
        print(f"Saving split {str(idx)} to {target_file}")
        torch.save({
            "source": source,
            "target": target
        }, target_file)
        
