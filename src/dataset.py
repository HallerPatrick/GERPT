import hashlib
import os
import string
import sys
from collections import Counter, defaultdict
from itertools import zip_longest
from operator import itemgetter
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from datasets import Dataset as HfDataset
from datasets import load_dataset as ld
from datasets.dataset_dict import DatasetDict
from datasets.fingerprint import Hasher
from datasets.load import load_from_disk
from nltk import ngrams as ngram_tokenizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from . import HF_CACHE_DICTIONARIES, HF_CACHE_TOKENIZED, USE_CACHE
from .data import local_dataset_mapper


def batch_collate(batch):
    # [ngram, seq_len, batch_size]
    source = torch.cat([torch.tensor(t["source"]).unsqueeze(-1) for t in batch], dim=-1)
    target = torch.cat([torch.tensor(t["target"]).unsqueeze(-1) for t in batch], dim=-1)
    return dict(source=source, target=target)


class GenericDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, cpus=1):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.cpus = cpus

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            collate_fn=batch_collate,
            drop_last=True,
            num_workers=self.cpus,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.batch_size,
            collate_fn=batch_collate,
            drop_last=True,
            num_workers=self.cpus,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.batch_size,
            collate_fn=batch_collate,
            drop_last=True,
            num_workers=self.cpus,
        )


# Cant pickle lambdas...
def zero():
    return 0


class Dictionary:
    def __init__(
        self, ngram: int, max_dict_size: int, unk_threshold: int, fallback: bool
    ):
        self.word2idx = {}
        self.idx2word = []
        self._marker_tokens = {}
        self.ngram_indexes = defaultdict(list)
        self.ngram = ngram
        self.max_dict_size = max_dict_size
        self.unk_threshold = unk_threshold
        self.fallback = fallback
        self.frequencies: Optional[Counter] = None
        self.total_n_tokens = defaultdict(zero)
        self.unk_n_tokens = defaultdict(zero)

        self.ngram2word2idx = {}
        self.ngram2idx2word = {}


        self.current_max_idx = 0

    def add_word(self, word):
        if word.startswith("<") and word.endswith(">"):
            if word not in self._marker_tokens:
                self._marker_tokens.append(word)

        if word not in self.word2idx:
            print(f"Adding new token: {repr(word)}")
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

        return self.word2idx[word]

    def add_ngram(self, word, ngram: int):

        if ngram not in self.ngram2idx2word:
            self.ngram2idx2word[ngram] = {self.current_max_idx: word}
            self.ngram2word2idx[ngram] = {word: self.current_max_idx}
        else:
            self.ngram2idx2word[ngram][self.current_max_idx] = word
            self.ngram2word2idx[ngram][word] = self.current_max_idx

        self.current_max_idx += 1

        return self.ngram2word2idx[ngram][word]

    def add_item(self, word):
        return self.add_word(word)

    def get_marker_tokens(self) -> List[str]:
        return self._marker_tokens

    def __len__(self):
        return self.current_max_idx

    def get_idx_for_item(self, item: str) -> int:
        """
        returns the ID of the string, otherwise 0
        :param item: string for which ID is requested
        :return: ID of string, otherwise 0
        """
        item = item.encode("utf-8")
        if item in self.word2idx.keys():
            return self.word2idx[item]
        else:
            return 0

    def get_idx_for_items(self, items: List[str]) -> List[int]:
        """
        returns the IDs for each item of the list of string, otherwise 0 if not found
        :param items: List of string for which IDs are requested
        :return: List of ID of strings
        """
        if not hasattr(self, "item2idx_not_encoded"):
            d = dict([(key, value) for key, value in self.word2idx.items()])
            self.item2idx_not_encoded = defaultdict(int, d)

        if not items:
            return []
        results = itemgetter(*items)(self.item2idx_not_encoded)
        if isinstance(results, int):
            return [results]
        return list(results)

    def get_items(self) -> List[str]:
        items = []
        for item in self.idx2word:
            items.append(item)
        return items

    def get_item_for_index(self, idx):
        for idxs in self.ngram2idx2word.values(): 
            if idx in idxs:
                return idxs[idx]
        print(idx)
        print(self.ngram2idx2word)
        exit()
        return None

    def save(self, savefile):
        import pickle

        with open(savefile, "wb") as f:
            mappings = {"idx2item": self.idx2word, "item2idx": self.word2idx}
            pickle.dump(mappings, f)

    def save_vocabulary(
        self,
        save_directory: Union[str, Path],
        vocab_file_name: str,
        ngram: int,
        filename_prefix: Optional[str] = None,
    ) -> str:

        if isinstance(save_directory, Path):
            save_directory = str(save_directory)

        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + vocab_file_name,
            )
        else:
            vocab_file = (
                filename_prefix + "-" if filename_prefix else ""
            ) + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            writer.write(str(ngram) + "\n")
            for token, token_index in sorted(
                self.word2idx.items(), key=lambda kv: kv[1]
            ):
                if index != token_index:
                    print(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index

                # TODO:Is this sound?
                if "\n" in token:
                    token = token.replace("\n", "\\n")

                writer.write(token + "\n")
                index += 1
        return vocab_file

    def tokenize_line(self, line: List[str]) -> dict:
        """

        line: List of chars

        h       e     l  l  o

        <s>h    he    el ll lo

        <s><s>h <s>he hel

        """

        ngram_sequences = []
        ngram_target_sequences = []
        min_length = sys.maxsize

        for n in range(1, self.ngram + 1):

            # Adding start offsets for all ngrams
            words = ["<start>" for _ in range(1, n)]
            words.extend(list(line))

            ids = []
            length = 0
            # print(f"Processed line: {words}")
            for c in words:

                ids.append(self.ngram2word2idx[n][c])
                length += 1

            seq = torch.tensor(ids).type(torch.int64).unsqueeze(dim=0)
            length = seq.size(1)

            if length < min_length:
                min_length = length

            # display_input_n_gram_sequences(seq, self)
            ngram_sequences.append(seq)
            s = self.shift_left(seq, n)
            # display_input_n_gram_sequences(s, self)
            ngram_target_sequences.append(s)

        sequence = torch.cat([torch.tensor(t[0][:min_length]).unsqueeze(0) for t in ngram_sequences])
        target = torch.cat([torch.tensor(t[0][:min_length]).unsqueeze(0) for t in ngram_target_sequences])

        return {"text": line, "source": sequence, "target": target}

    def shift_left(self, t: torch.Tensor, shifts) -> torch.Tensor:
        """
        "hello world"

        1. "h"   "e"   "l"   "l"   "o"
        2. "he"  "el"  "ll"  "lo"  "o "  " w"
        3. "hel" "ell" "llo" "lo " "o w" " wo"



        1. "e"   "l"   "l"   "o"   " "
        2. "ll"  "lo"  "o "  " w"  :offset 2
        3. "lo " "o w" " wo"       :offset 3


        Shifts have to be applied ngram-time for correct target matching
        """
        st = torch.roll(t, -shifts, 1)

        st[0][-1] = self.ngram2word2idx[1]["<pad>"]
        for i in range(1, shifts + 1):
            st[0][-i] = self.ngram2word2idx[i]["<pad>"]
        return st

    def create_weight_tensor(self) -> Optional[list]:
        # if not self.frequencies:
        #     return

        t = torch.ones(len(self))

        normed_weights = t
    
        for marker in self.get_marker_tokens():
            normed_weights[marker] = 0
        
        return normed_weights


def get_dictionary_cache() -> Path:
    path = Path(HF_CACHE_DICTIONARIES)

    if not path.exists():
        path.mkdir()

    return path


def get_tokenized_cache() -> Path:
    path = Path(HF_CACHE_TOKENIZED)

    if not path.exists():
        try:
            path.mkdir()
        except FileNotFoundError as e:
            print(e)

    return path


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def load_tokenized_dataset(
    bptt: int,
    ngram: int,
    max_dict_size: int,
    unk_threshold: int,
    fallback: bool,
    num_proc: int,
    *args,
    **kwargs,
) -> Tuple[Dataset, Dictionary]:
    """🤗"""

    # Check if we have a local config for local dataset
    if args[0] == "text" and args[1] in local_dataset_mapper:
        dataset = ld("text", data_files=local_dataset_mapper[args[1]])
    else:
        # Load the datasets from huggingface
        dataset = ld(*args, **kwargs)

    # Load according dictionary for dataset
    dictionary = load_dictionary_from_hf(
        dataset, ngram, max_dict_size, unk_threshold, fallback
    )

    print(dictionary.ngram2word2idx)

    # Check if we have a cached tokenized version of the dataset already in the huggingface cache

    hash_value = hashlib.md5(
        f"{Hasher.hash(dataset)}{Hasher.hash(dictionary)}".encode()
    ).hexdigest()
    hashed_file = get_tokenized_cache() / hash_value

    if hashed_file.exists() and USE_CACHE:
        print(f"Loading cached processed tokenized dataset at {hashed_file.resolve()}")
        return load_from_disk(hashed_file), dictionary

    print("Preprocess dataset...")
    train = [x["text"] for x in dataset["train"]]
    test = [x["text"] for x in dataset["test"]] if "test" in dataset else []
    valid = (
        [x["text"] for x in dataset["validation"]] if "validation" in dataset else []
    )

    # train = map(lambda x: x["text"], dataset["train"])
    # test = map(lambda x: x["text"], dataset["test"])
    # valid = map(lambda x: x["text"], dataset["validation"])

    train = "\n".join(train)
    test = "\n".join(test)
    valid = "\n".join(valid)

    # # Make continuous sequence
    # train = reduce(concat, train)
    # test = reduce(concat, test)
    # valid = reduce(concat, valid)

    # Divide sequence into bptt
    dataset = DatasetDict(
        {
            "train": HfDataset.from_dict({"text": list(grouper(train, bptt, " "))}),
            "test": HfDataset.from_dict({"text": list(grouper(test, bptt, " "))}),
            "validation": HfDataset.from_dict(
                {"text": list(grouper(valid, bptt, " "))}
            ),
        }
    )

    print("Tokenize dataset...")
    tokenized_dataset = dataset.map(
        lambda x: dictionary.tokenize_line(x["text"]),
        load_from_cache_file=USE_CACHE,
        num_proc=num_proc,
    )

    print(f"Saving tokenized dataset at {hashed_file.resolve()}")
    try:
        tokenized_dataset.save_to_disk(str(hashed_file.resolve()))
    except OSError:
        print("Failed to save... 😢")

    return tokenized_dataset, dictionary


def load_dictionary_from_hf(
    source: Dataset, ngrams: int, max_dict_size: int, unk_threshold: int, fallback: bool
) -> Dictionary:

    # Hash the combination of dataset and configs
    hash_value = hashlib.md5(
        f"{Hasher.hash(source)}{ngrams}{max_dict_size}{unk_threshold}".encode()
    ).hexdigest()

    hash_file = get_dictionary_cache() / hash_value

    if hash_file.exists() and USE_CACHE:
        print(f"Loading cached processed dictionary at {hash_file.resolve()}")
        return torch.load(hash_file)

    dictionary = Dictionary(ngrams, max_dict_size, unk_threshold, fallback)

    all_chars = []

    for train_split in ["train", "test", "validation"]:

        try:
            split = source[train_split]
        except KeyError:
            continue

        lines = split["text"]

        uniq_chars = set("\n".join(lines))

        all_chars.extend(list(uniq_chars))

    all_chars = set(all_chars)
        
    for n_gram in range(1, ngrams + 1):
        start_idx = dictionary.add_ngram("<start>", n_gram)
        pad_idx = dictionary.add_ngram("<pad>", n_gram)

        if n_gram not in dictionary._marker_tokens:
            dictionary._marker_tokens[n_gram] = [start_idx, pad_idx]

        for char in all_chars:
            dictionary.add_ngram(char, n_gram)

    print(f"Saving dictionary at {hash_file}...")
    torch.save(dictionary, hash_file)

    return dictionary


def remove_marker_tokens(token, dictionary):
    """Due to some str length comparison to determine what n-gram the token
    is. We replace the marker tokens, with a single char, for easy comparison
    """
    for marker in dictionary.get_marker_tokens():
        token = token.replace(marker, "i")

    return token
