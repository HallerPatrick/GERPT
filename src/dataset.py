import numpy as np
import pytorch_lightning as pl
import torch
import functools
import itertools

from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import ConcatDataset, IterableDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from .data import batchify
from .processor import Processor


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


class SplitTextDataset(Dataset):
    def __init__(self, ds_iterator, batch_size, bptt_size, pad_tokens) -> None:
        self.bptt = bptt_size
        self.batch_size = batch_size

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
        self.train = TextDataset(
            self.dataset["train"], self.batch_size, self.bptt_size, self.pad_tokens
        )
        self.test = TextDataset(
            self.dataset["test"], self.batch_size, self.bptt_size, self.pad_tokens
        )
        self.valid = TextDataset(
            self.dataset["valid"], self.batch_size, self.bptt_size, self.pad_tokens
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


class ShardedDataModule(pl.LightningDataModule):
    def __init__(
        self, train_ds, valid_ds, test_ds, batch_size, bptt_size, pad_tokens, cpus=1
    ):
        super().__init__()
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.bptt_size = bptt_size
        self.pad_tokens = pad_tokens
        self.cpus = cpus

    def setup(self, stage):
        self.prepare_data()

    def prepare_data(self) -> None:
        self.train = TextDataset(
            self.train_ds, self.batch_size, self.bptt_size, self.pad_tokens
        )
        self.valid = TextDataset(
            self.valid_ds, self.batch_size, self.bptt_size, self.pad_tokens
        )
        self.test = TextDataset(
            self.test_ds, self.batch_size, self.bptt_size, self.pad_tokens
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


class SplitIterator:
    def __init__(self, split, batch_size, bptt_size, pad_tokens):
        self.split = split
        self.batch_size = batch_size
        self.bptt = bptt_size
        self.pad_tokens = pad_tokens

        self.current_batch = 0
        self.idx = 0

    def __iter__(self):
        self.inputs, self.nbatch = batchify(
            self.split["source"], self.batch_size, self.bptt
        )
        self.target, _ = batchify(self.split["target"], self.batch_size, self.bptt)
        return self

    def __len__(self):
        return self.nbatch * self.batch_size

    def __next__(self):
        idx = self.idx // self.batch_size
        start_idx = idx * self.bptt
        end_idx = (idx + 1) * self.bptt

        if end_idx > self.inputs.shape[1]:
            raise StopIteration

        source = self.inputs[:, start_idx:end_idx, self.current_batch]

        # Targets already shifted
        target = self.target[:, start_idx:end_idx, self.current_batch]

        # We dont need it?
        # target = self._pad_target(target)

        self.current_batch += 1

        if self.current_batch == self.inputs.shape[2]:
            self.current_batch = 0

        assert isinstance(source, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        self.idx += 1

        return source, target


class TextIteratorDataset(torch.utils.data.IterableDataset):
    def __init__(self, ds_iterator, batch_size, bptt_size, pad_tokens, num_workers):
        super(TextIteratorDataset).__init__()
        self.ds_iterator = ds_iterator
        self.batch_size = batch_size
        self.bptt = bptt_size
        self.pad_tokens = pad_tokens

    def __len__(self):
        return len(self.ds_iterator)

    def __iter__(self):
        for split in self.ds_iterator:
            yield from iter(SplitIterator(
                split["train"], self.batch_size, self.bptt, self.pad_tokens
            ))

class SplitDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, bptt_size, pad_tokens, cpus=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.bptt_size = bptt_size
        self.pad_tokens = pad_tokens
        self.cpus = cpus

    def train_dataloader(self):
        print("Loading train data")
        return DataLoader(
            TextIteratorDataset(
                Processor.from_strategy("split").read_dataset(self.data_dir),
                self.batch_size,
                self.bptt_size,
                self.pad_tokens,
                self.cpus,
            ),
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.cpus,
            pin_memory=True,
        )
