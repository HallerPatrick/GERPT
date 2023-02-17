import copy
import multiprocessing
import pprint as pp
import sys
from collections import Counter, OrderedDict, defaultdict
from functools import lru_cache, partial
from typing import Iterator, List, Optional, Union, Tuple

import flair
from flair.embeddings import FlairEmbeddings
import nltk
import numpy as np
import pyarrow
import torch
from tqdm import tqdm

from datasets import Dataset
from src import utils


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = iter(reader.readlines())

    try:
        ngrams, ngme_type = next(tokens).strip().split(" ")
        ngrams = int(ngrams)
    except:
        print("Could not determine ngram of tokenizer in vocab file")
        exit(-1)

    for index, token in enumerate(tokens):
        token = token.rstrip("\n")

        ngram = int(token[0])
        token = token[2:]

        if "\\n" in token:
            token = token.replace("\\n", "\n")

        if ngram not in vocab:
            vocab[ngram] = {token: index}
        else:
            vocab[ngram][token] = index

    return ngrams, ngme_type, vocab


class Dictionary:
    def __init__(
        self,
        ngram: int,
        max_dict_size: int,
        ngme: str,
        packed: bool = False,
    ):
        self._marker_tokens = {}
        self.ngram_indexes = defaultdict(list)
        self.ngram = ngram
        self.max_dict_size = max_dict_size
        self.frequencies: Counter = Counter()
        self.pad_tokens = {}

        self.ngram2word2idx = {}
        self.ngram2idx2word = {}

        self.current_max_idx = 0
        self.ngme = ngme
        self.packed = packed

    @classmethod
    def load_from_file(cls, filename: str):
        ngram, ngme_type, vocab = load_vocab(filename)

        # TODO: Not sufficient, save more meta data in dict file
        dictionary = cls(ngram, 0, ngme_type, False)

        for ngram in vocab:
            for token in vocab[ngram]:
                dictionary.add_ngram(token, ngram)

        return dictionary

    @classmethod
    def build_from_dataset(
        cls,
        dataset: Union[Dataset, Iterator[Dataset]],
        ngram: int,
        max_dict_size: int,
        ngme: str,
        packed: bool,
        num_proc: int,
    ) -> Tuple["Dictionary", List[Dataset]]:
        """Builds a dictionary from a dataset.

        Note: We return the processed dataset as well, because the generator consumes the
        dataset. If we would not return it, we would have to recreate the
        iterator.
        """
        dictionary = cls(ngram, max_dict_size, ngme, packed)

        # Populate dictionary
        if ngme == "compositional":
            print("Populate compositional dictionary...", end="")
            dictionary.populate_compositional()
            print("Done")
        elif ngme == "explicit":
            print("Populate explicit dictionary...", end="")
            # If dataset is an generator, we need to copy it
            if isinstance(dataset, Iterator):
                processed_dataset = []
                for dataset_for_dict in dataset:
                    dictionary.populate_explicit(dataset_for_dict["train"]["text"], num_proc)
                    processed_dataset.append(dataset_for_dict)
                dataset = processed_dataset
            else:
                dictionary.populate_explicit(dataset["train"]["text"], num_proc)

            print("Done")
        else:
            raise ValueError("Unknown ngme type")

        # Check if we need to apply unking
        if dictionary.max_dict_size == 0:
            dictionary.max_dict_size = len(dictionary)

        if dictionary.max_dict_size < len(dictionary):
            print("Apply unking...", end="")
            dictionary.unking()
            print("Done")

        # Check if all unigrams were indexed first and all idx are consecutive
        assert list(dictionary.ngram2idx2word[1].keys()) == list(
            range(0, len(dictionary.ngram2idx2word[1]))
        )

        return dictionary, dataset

    def populate_compositional(self):
        """Populate dictionary with compositional n-gram tokens. We base it on the unigram tokens of 'news-forward'"""
        self.ngme = "compositional"
        unigram_tokens = get_unigram_tokens()
        for n_gram in range(1, self.ngram + 1):
            self.add_ngram("<start>", n_gram)
            self.add_ngram("<pad>", n_gram)
            self.add_ngram("<unk>", n_gram)

            for token in unigram_tokens:
                self.add_ngram(token, n_gram)

    def populate_explicit(self, rows: List[str], num_proc: int):
        """Populate dictionary from a list of strings with explicit n-gram tokens"""
        self.ngme = "explicit"

        # Guarantee that all unigram tokens are indexed first
        # Uni-gram tokens
        for token in get_unigram_tokens():
            self.add_ngram(token, 1)

        # Add new n-gram token only if all uni-gram parts exist
        for n in range(1, self.ngram + 1):
            start_idx = self.add_ngram("<start>", n)
            pad_idx = self.add_ngram("<pad>", n)
            unk_idx = self.add_ngram("<unk>", n)
            self.add_ngram(" " * n, n)
            self._marker_tokens[n] = [start_idx, pad_idx, unk_idx]

        ngram_list = list(range(1, self.ngram + 1))
        if num_proc > 1:
            print(f"Run dictionary collection with {num_proc} workers")
            with multiprocessing.Pool(num_proc) as pool:
                for ngram_tokens in tqdm(
                    pool.map(
                        partial(add_ngrams_from_text, ngrams=ngram_list),
                        rows,
                        chunksize=25,
                    )
                ):
                    for n, tokens in ngram_tokens.items():
                        self.add_ngrams(tokens, n)

        else:
            for line in rows:
                for n, tokens in add_ngrams_from_text(line, ngram_list).items():
                    self.add_ngrams(tokens, n)

    def get_all_tokens(self):
        for ngram in range(1, self.ngram + 1):
            for idx, token in self.ngram2idx2word[ngram].items():
                yield idx, token

    def get_ngram_order(self, idx) -> int:
        for ngram, idx2word in self.ngram2idx2word.items():
            if idx in idx2word:
                return ngram
        return -1

    def add_ngram(self, word, ngram: int):
        self.frequencies.update({word: 1})

        if ngram not in self.ngram2idx2word:
            self.ngram2idx2word[ngram] = {self.current_max_idx: word}
            self.ngram2word2idx[ngram] = {word: self.current_max_idx}
            self.current_max_idx += 1
        else:
            if word not in self.ngram2word2idx[ngram]:
                self.ngram2idx2word[ngram][self.current_max_idx] = word
                self.ngram2word2idx[ngram][word] = self.current_max_idx
                self.current_max_idx += 1

        return self.ngram2word2idx[ngram][word]

    def add_ngrams(self, words: List[str], ngram: int):
        self.frequencies.update(Counter(words))

        for word in words:
            if ngram not in self.ngram2idx2word:
                self.ngram2idx2word[ngram] = {self.current_max_idx: word}
                self.ngram2word2idx[ngram] = {word: self.current_max_idx}
                self.current_max_idx += 1
            else:
                if word not in self.ngram2word2idx[ngram]:
                    self.ngram2idx2word[ngram][self.current_max_idx] = word
                    self.ngram2word2idx[ngram][word] = self.current_max_idx
                    self.current_max_idx += 1

    def unking(
        self, new_max_dict_size: Optional[int] = None, ngrams: Optional[int] = None
    ):
        """Trim the dictionary size to `new_max_dict_size` or self.max_dict_size.

        If `ngram` is set, apply unking after subsetting ngram vocabs up to argument.
        """

        max_dict_size = new_max_dict_size if new_max_dict_size else self.max_dict_size

        if ngrams:
            sub_frequencies = {}
            for token, freq in self.frequencies.items():
                n_order = self.token_to_n_order(token)

                if n_order <= ngrams:
                    sub_frequencies.update({token: freq})

            frequencies = Counter(sub_frequencies)
        else:
            frequencies = self.frequencies
            ngrams = self.ngram

        candidates = list(map(lambda x: x[0], frequencies.most_common(max_dict_size)))

        dictionary = Dictionary(ngrams, max_dict_size, self.ngme)

        for ngram in self.ngram2idx2word.keys():
            for token, ngram_idx in self.ngram2word2idx[ngram].items():
                # Add token if:
                # 1. Token occurs often enough
                # 2. Is a marker token
                # 3. Is unigram
                if (
                    token in candidates
                    or ngram_idx in self._marker_tokens[ngram]
                    # or ngram == 1 TODO: Do we really need to ensure all unigrams are in?
                ):
                    dictionary.add_ngram(token, ngram)

        dictionary.frequencies = frequencies

        return dictionary

    def get_marker_tokens(self) -> Iterator[int]:
        for ngram in self._marker_tokens:
            for token_idx in self._marker_tokens[ngram]:
                yield token_idx

    def __len__(self):
        return self.current_max_idx

    def get_item_for_index(self, idx):
        for idxs in self.ngram2idx2word.values():
            if idx in idxs:
                return idxs[idx]
        return None

    def save_vocabulary(
        self,
        vocab_file: str,
        ngram: int,
    ) -> str:
        index = 0

        with open(vocab_file, "w", encoding="utf-8") as writer:
            writer.write(str(ngram) + " " + self.ngme + "\n")

            for ngram in range(1, self.ngram + 1):
                for idx, token in self.ngram2idx2word[ngram].items():
                    if index != idx:
                        print(
                            f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                            " Please check that the vocabulary is not corrupted!"
                        )
                        index = idx

                    # TODO:Is this sound?
                    if "\n" in token:
                        token = token.replace("\n", "\\n")

                    writer.write(str(ngram) + " " + token + "\n")
                    index += 1
        return vocab_file

    def tokenize_line(
        self,
        line: Union[str, List[str]],
        id_type=torch.int64,
        return_tensor: Optional[str] = None,
        with_text: bool = True,
    ) -> dict:
        if self.ngme == "explicit":
            return self._tokenize_line_explicit(line, id_type, return_tensor, with_text)
        elif self.ngme == "compositional":
            return self._tokenize_line_compositional(line, id_type, return_tensor, with_text)
        else:
            raise ValueError("UNKOWN NGME APPROACH")

    @lru_cache
    def _special_tokens(self) -> List[str]:
        return [
            self.ngram2idx2word[1][special_token_id]
            for special_token_id in self._marker_tokens[1]
        ]

    def _tokenize_line_explicit(
        self,
        line: Union[str, List[str]],
        id_type,
        return_tensor: str = "list",
        with_text=True,
    ):
        ngram_sequences = []
        ngram_target_sequences = []
        min_length = sys.maxsize

        for n in range(1, self.ngram + 1):
            # Adding start offsets for all ngrams
            words = ["<start>" for _ in range(1, n)]
            words.extend(list(line))

            ids = []
            length = 0
            for i, word in enumerate(nltk.ngrams(words, n)):
                if "<start>" in word:
                    word = [w for w in list(word) if w != "<start>"]

                try:
                    ids.append(self.ngram2word2idx[n]["".join(word)])
                except KeyError:
                    ids.append(self.ngram2word2idx[n]["<unk>"])

                length += 1

            seq = torch.tensor(ids).type(id_type).unsqueeze(dim=0)

            # length = len(seq)
            length = seq.size(1)

            if length < min_length:
                min_length = length

            ngram_sequences.append(seq)

            try:
                s = self.shift_left(seq, n)
            except IndexError:
                s = seq
            ngram_target_sequences.append(s)

        sequence = torch.cat([t[:min_length] for t in ngram_sequences])
        target = torch.cat([t[:min_length] for t in ngram_target_sequences])

        if self.packed:
            sequence = utils.pack_tensor(sequence)
            target = utils.pack_tensor(target)

        if with_text:
            return_value = {"text": line, "source": sequence, "target": target}
        else:
            return_value = {"source": sequence, "target": target}

        return self.convert_return_value(return_value, return_tensor)

    @staticmethod
    def convert_return_value(return_dict, return_tensor):
        if return_tensor == "pt":
            return return_dict

        # Return per default list
        if not return_tensor:
            return_dict["source"] = return_dict["source"].tolist()
            return_dict["target"] = return_dict["target"].tolist()
            return return_dict

        return_dict["source"] = Dictionary._to_tensor(
            return_dict["source"], return_tensor
        )
        return_dict["target"] = Dictionary._to_tensor(
            return_dict["target"], return_tensor
        )

        if return_tensor == "arrow":
            if "text" in return_dict:
                del return_dict["text"]
                # return_dict["text"] = list(return_dict["text"])
            return pyarrow.Table.from_arrays(
                [return_dict["source"], return_dict["target"]], ["source", "target"]
            )

        return return_dict

    @staticmethod
    def _to_tensor(tensor, t_type):
        if t_type == "list":
            return tensor.tolist()
        if t_type == "np":
            return np.array(tensor)
        if t_type == "pt":
            return torch.tensor(tensor)
        if t_type == "arrow":
            return pyarrow.array(tensor.tolist())

        return list(tensor)

    def _tokenize_line_compositional(
        self, line: Union[str, List[str]], id_type, return_tensor, with_text=True
    ):
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
            for c in words:
                try:
                    ids.append(self.ngram2word2idx[n][c])
                except KeyError:
                    ids.append(self.ngram2word2idx[n]["<unk>"])
                length += 1

            seq = torch.tensor(ids).type(id_type).unsqueeze(dim=0)
            length = seq.size(1)

            if length < min_length:
                min_length = length

            # display_input_n_gram_sequences(seq, self)
            ngram_sequences.append(seq)
            s = self.shift_left(seq, n)
            # display_input_n_gram_sequences(s, self)
            ngram_target_sequences.append(s)

        sequence = torch.cat(
            [t[0][:min_length].clone().detach().unsqueeze(0) for t in ngram_sequences]
        )
        target = torch.cat(
            [
                t[0][:min_length].clone().detach().unsqueeze(0)
                for t in ngram_target_sequences
            ]
        )

        if with_text:
            return_value = {"text": line, "source": sequence, "target": target}
        else:
            return_value = {"source": sequence, "target": target}

        if return_tensor == "arrow":
            print("Table")
            return pyarrow.Table.from_pydict(return_value)

        return return_value

    def shift_left(self, t: Union[List, torch.Tensor], shifts) -> torch.Tensor:
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
        if isinstance(t, list):
            t = torch.tensor(t)

        # Roll sequences now
        st = torch.roll(t, -shifts, 1)

        # Apply padding later!
        # for i in range(1, shifts + 1):
        #     st[0][-i] = self.ngram2word2idx[i]["<pad>"]
        return st

    def create_weight_tensor(
        self, unigram_ppl: bool, weighted_loss: bool = True
    ) -> torch.Tensor:
        unked_freqs = self.frequencies.most_common(self.max_dict_size)

        if unigram_ppl:
            t = torch.ones(len(self.ngram2idx2word[1]))
        else:
            t = torch.ones(len(self))

        if not unigram_ppl and weighted_loss:
            for token, freq in unked_freqs:
                t[self.ngram2word2idx[self.token_to_n_order(token)][token]] = freq

            max_t = max(t)

            normed_weights = torch.tensor([(1 - (x / (max_t + 1))).item() for x in t])
        else:
            normed_weights = t

        for marker in self.get_marker_tokens():
            if (
                marker < len(normed_weights)
                and self.get_item_for_index(marker) != "<eod>"
            ):
                normed_weights[marker] = 0

        return normed_weights

    def token_to_n_order(self, token: str) -> int:
        for n in self.ngram2word2idx:
            if token in self.ngram2word2idx[n]:
                return n

        return 0

    def print_sequence(self, seq, ngram):
        """docstring for print_sequence"""
        collected_tokens = self._get_char_sequence(seq, ngram)
        print(f"[{', '.join(collected_tokens)}]")

    def _get_char_sequence(self, seq, ngram):
        collected_tokens = []

        for token in seq:
            if not isinstance(token, int):
                token = token.item()
            collected_tokens.append(self.ngram2idx2word[ngram][token])

        return collected_tokens

    def print_batch(self, batch):
        seqs = []
        for batch_idx in range(batch.size(2)):
            for n_idx in range(batch.size(0)):
                seqs.append(
                    self._get_char_sequence(batch[n_idx, :, batch_idx], n_idx + 1)
                )

        for seq in seqs:
            print(seq)

def collect_ngrams(line, n):
    return ["".join(ngram) for ngram in nltk.ngrams(line, n)]

def add_ngrams_from_text(text: str, ngrams: List[int]):
    return {ngram: collect_ngrams(text, ngram) for ngram in ngrams}


def get_unigram_tokens() -> List[str]:
    flair_device = flair.device
    flair.device = "cpu"

    # Using unigrams from flair as base
    e = FlairEmbeddings("news-forward")
    flair.device = flair_device
    return list(e.lm.dictionary.item2idx_not_encoded.keys())
