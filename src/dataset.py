import gc

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_dataset as ld
from datasets.dataset_dict import DatasetDict
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from .data import batchify, local_dataset_mapper


class TextDataset(Dataset):
    def __init__(self, ds, batch_size, bptt_size, pad_tokens) -> None:
        self.bptt = bptt_size
        self.batch_size = batch_size
        print(ds["source"].shape)
        self.inputs, self.nbatch = batchify(ds["source"], batch_size, bptt_size)
        # print(self.inputs)
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
        self.train = TextDataset(self.dataset["train"], self.batch_size, self.bptt_size, self.pad_tokens)
        self.test = TextDataset(self.dataset["test"], self.batch_size, self.bptt_size, self.pad_tokens)
        self.valid = TextDataset(self.dataset["valid"], self.batch_size, self.bptt_size, self.pad_tokens)


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

class ShardedDataModule(GenericDataModule):
    def setup(self, stage):

        self.train = ConcatDataset(
            [
                TextDataset(data, self.batch_size, self.bptt_size, self.pad_tokens)
                for data in self.dataset["train"]
            ]
        )
        print(len(self.train))
        self.test = ConcatDataset(
            [
                TextDataset(data, self.batch_size, self.bptt_size, self.pad_tokens)
                for data in self.dataset["test"]

            ]
        )
        print(len(self.test))
        self.valid = ConcatDataset(
            [
                TextDataset(data, self.batch_size, self.bptt_size, self.pad_tokens)
                for data in self.dataset["validation"]
            ]
        )
        print(len(self.valid))

def get_text(x):
    return x["text"]


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

