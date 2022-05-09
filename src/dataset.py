import hashlib
import sys
from collections import Counter, defaultdict
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple
from datasets.load import load_from_disk

import torch
from colorama.ansi import Fore
from datasets import load_dataset as ld
from datasets.fingerprint import Hasher
from nltk import ngrams as ngram_tokenizer
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from . import HF_CACHE_DICTIONARIES, HF_CACHE_TOKENIZED




class Dictionary:
    def __init__(
        self, ngram: int, max_dict_size: int, unk_threshold: int, fallback: bool
    ):
        self.word2idx = {}
        self.idx2word = []
        self._marker_tokens = []
        self.ngram_indexes = defaultdict(list)
        self.ngram = ngram
        self.max_dict_size = max_dict_size
        self.unk_threshold = unk_threshold
        self.fallback = fallback

    def add_word(self, word):
        if word.startswith("<") and word.endswith(">"):
            if word not in self._marker_tokens:
                self._marker_tokens.append(word)

        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def add_item(self, word):
        return self.add_word(word)

    def get_marker_tokens(self) -> List[str]:
        return self._marker_tokens

    def __len__(self):
        return len(self.idx2word)

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
        return self.idx2word[idx]

    def save(self, savefile):
        import pickle

        with open(savefile, "wb") as f:
            mappings = {"idx2item": self.idx2word, "item2idx": self.word2idx}
            pickle.dump(mappings, f)

    def tokenize_line(self, line: str) -> dict:
        sequence = []
        min_length = sys.maxsize

        for n in range(1, self.ngram + 1):

            # Adding start offsets for all ngrams
            words = ["<start>" for _ in range(1, n)]
            words.extend(list(line))

            # TODO: Do we need that?
            # if not self.otf:
            #     words.append("<eos>")

            ids = []
            length = 0
            for i, word in enumerate(ngram_tokenizer(words, n)):
                try:
                    ids.append(self.word2idx["".join(word)])
                except KeyError:
                    # Fall back on n-1 gram if possible
                    if self.fallback and word[1:] in self.word2idx:
                        ids.append(self.word2idx[word])
                    else:
                        ids.append(self.word2idx[f"<{n}-UNK>"])
                length += 1

            sequence.append(torch.tensor(ids).type(torch.int64))

        seq = torch.cat(sequence).unsqueeze(dim=0)
        length = seq.size(1)

        if length < min_length:
            min_length = length

        sequence = torch.cat([t[:min_length] for t in sequence])
        return {"text": line, "tensor": sequence}


def get_dictionary_cache() -> Path:
    path = Path(HF_CACHE_DICTIONARIES)

    if not path.exists():
        path.mkdir()

    return path

def get_tokenized_cache() -> Path:
    path = Path(HF_CACHE_TOKENIZED)
    if not path.exists():
        path.mkdir()

    return path

def load_tokenized_dataset(
    ngram, max_dict_size, unk_threshold, fallback, *args, **kwargs
) -> Tuple[Dataset, Dictionary]:
    """🤗"""

    # Load the datasets from huggingface
    dataset = ld(*args, **kwargs)

    # Load according dictionary for dataset
    dictionary = load_dictionary_from_hf(
        dataset, ngram, max_dict_size, unk_threshold, fallback
    )

    # Check if we have a cached tokenized version of the dataset already in the huggingface cache

    hash_value = hashlib.md5(f"{Hasher.hash(dataset)}{Hasher.hash(dictionary)}".encode()).hexdigest()
    hashed_file = get_tokenized_cache() / hash_value

    if hashed_file.exists():
        print(f"Loading cached processed tokenized dataset at {hashed_file.resolve()}")
        return load_from_disk(hashed_file), dictionary
    
    tokenized_dataset = dataset.map(lambda x: dictionary.tokenize_line(x["text"]))
    print(f"Saving tokenized dataset at {hashed_file.resolve()}")
    tokenized_dataset.save_to_disk(str(hashed_file.resolve()))

    return tokenized_dataset, dictionary

def load_dictionary_from_hf(
    source: Dataset, ngrams: int, max_dict_size: int, unk_threshold: int, fallback: bool
) -> Dictionary:

    # Hash the combination of dataset and configs
    hash_value = hashlib.md5(
        f"{Hasher.hash(source)}{ngrams}{max_dict_size}{unk_threshold}".encode()
    ).hexdigest()

    hash_file = get_dictionary_cache() / hash_value

    if hash_file.exists():
        print(f"Loading cached processed dictionary at {hash_file.resolve()}")
        return torch.load(hash_file)

    dictionary = Dictionary(ngrams, max_dict_size, unk_threshold, fallback)
    frequencies = Counter()

    for train_split in ["train", "test", "validation"]:
        split = source[train_split]
        lines = split["text"]

        split_frequency = Counter()

        for line in tqdm(
            lines,
            desc=f"Setup dictionary from {Fore.MAGENTA}{train_split}{Fore.RESET} split",
        ):
            chars = ["<start>" for _ in range(1, ngrams)] + list(line) + ["<eos>"]
            for i in range(1, ngrams + 1):
                # Add UNK token for ngram
                n_unk_token = f"<{i}-UNK>"

                unk_idx = dictionary.add_word(n_unk_token)

                if unk_idx not in dictionary.ngram_indexes[i]:
                    dictionary.ngram_indexes[i].append(unk_idx)

                for ngram in ngram_tokenizer(chars, i):
                    split_frequency["".join(ngram)] += 1

        frequencies.update(split_frequency)

        dictionary.add_word("<start>")
        dictionary.add_word("<eos>")

        if max_dict_size > 0:
            for token, _ in frequencies.most_common(max_dict_size):
                sanit_token = remove_marker_tokens(token, dictionary)
                idx = dictionary.add_word(token)
                if idx not in dictionary.ngram_indexes[len(sanit_token)]:
                    dictionary.ngram_indexes[len(sanit_token)].append(idx)
        else:
            for token, freq in frequencies.items():
                if freq > unk_threshold or freq == -1:
                    sanit_token = remove_marker_tokens(token, dictionary)
                    idx = dictionary.add_word(token)
                    if idx not in dictionary.ngram_indexes[len(sanit_token)]:
                        dictionary.ngram_indexes[len(sanit_token)].append(idx)

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
