import json
import sys
from collections import Counter, defaultdict
from typing import Iterator, List, Optional, Tuple, Union

import flair
import nltk
import numpy as np
import pyarrow
import torch
from datasets import Dataset
from flair.embeddings import FlairEmbeddings

from src import utils


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    return json.load(open(vocab_file, "r", encoding="utf-8"))


class Dictionary:
    def __init__(
        self,
        ngram: int,
        max_dict_size: int,
        ngme: str,
        packed: bool = False,
    ):
        self.ngram = ngram
        self.max_dict_size = max_dict_size
        self.ngme = ngme
        self.packed = packed

        self._marker_tokens = defaultdict(list)
        self.frequencies: Counter = Counter()

        self.ngram2word2idx = {}
        self.ngram2idx2word = {}

        self.current_max_idx = 0

    @classmethod
    def load_from_file(cls, filename: str):
        """Loads a dictionary from a file."""
        vocab_file = load_vocab(filename)

        ngram = vocab_file["ngram"]
        ngme_type = vocab_file["ngme"]

        # TODO: Not sufficient, save more meta data in dict file
        dictionary = cls(ngram, len(vocab_file["vocab"]), ngme_type, False)

        for token in vocab_file["vocab"]:
            dictionary.add_ngram(token["token"], token["ngram"])
            dictionary.frequencies.update({token["token"]: token["frequency"]})

        return dictionary

    @classmethod
    def build_from_dataset(
        cls,
        dataset: Union[Dataset, Iterator[Dataset]],
        ngram: int,
        max_dict_size: int,
        ngme: str,
        packed: bool,
    ) -> Tuple["Dictionary", List[Dataset]]:
        """Builds a dictionary from a dataset.

        Note: We return the processed dataset as well, because the generator consumes the
        dataset. If we would not return it, we would have to recreate the
        iterator.
        """
        dictionary = cls(ngram, max_dict_size, ngme, packed)

        # Populate dictionary
        if ngme == "compositional":
            dictionary.populate_compositional()
        elif ngme == "explicit":
            # If dataset is an generator, we need to copy it
            if isinstance(dataset, Iterator):
                processed_dataset = []
                for dataset_for_dict in dataset:
                    dictionary.populate_explicit(dataset_for_dict["train"]["text"])
                    processed_dataset.append(dataset_for_dict)
                dataset = processed_dataset
            else:
                dictionary.populate_explicit(dataset["train"]["text"])
        else:
            raise ValueError(f"Unknown ngme type: {ngme}")

        # Check if we need to apply unking
        if dictionary.max_dict_size == 0:
            dictionary.max_dict_size = len(dictionary)

        if dictionary.max_dict_size < len(dictionary):
            print("Apply unking...", end="")
            dictionary = dictionary.unking()
            print("Done")

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

    def populate_explicit(self, rows: List[str]):
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

        # TODO: Rather expensive
        for n in ngram_list:
            for line in rows:
                tokens = collect_ngrams(line, n)
                self.add_ngrams(tokens, n)

    def get_all_tokens(self):
        """Returns all tokens in the dictionary."""
        for ngram in range(1, self.ngram + 1):
            for idx, token in self.ngram2idx2word[ngram].items():
                yield idx, token

    def get_ngram_order(self, idx) -> int:
        """Returns the n-gram order of a token."""
        for ngram, idx2word in self.ngram2idx2word.items():
            if idx in idx2word:
                return ngram
        return -1

    def add_ngram(self, word, ngram: int) -> int:
        """Add a new n-gram token to the dictionary."""
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
        """Add a list of n-gram tokens to the dictionary."""
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

    def _calculate_ngram_order_dict_size(self, ngrams: int, max_dict_size: Optional[int] = None) -> int:
        """Calculate the number of tokens for a given ngram order"""

        if not max_dict_size:
            max_dict_size = self.max_dict_size

        total_occurences = []

        for ngram in range(1, ngrams + 1):
            occurence = 0
            for word, occurences in self.frequencies.items():
                if word in self.ngram2word2idx[ngram]:
                    occurence += 1
            total_occurences.append(occurence)

        total_occurences = np.array(total_occurences)

        rel_occurences = total_occurences / total_occurences.sum()
        new_occurences = rel_occurences * (200 / rel_occurences[0])
        return new_occurences.astype(int)

    def _remove_whitespace_tokens(self):
        """Remove ngram tokens that contain whitespaces"""
        # Ignore ungiram
        for ngram in range(2, self.ngram + 1):
            for idx, token in self.ngram2idx2word[ngram].copy().items():
                if " " in token:
                    self.ngram2idx2word[ngram].pop(idx)
                    self.ngram2word2idx[ngram].pop(token)
                    self.current_max_idx -= 1
                    del self.frequencies[token]

    def unking(
        self, new_max_dict_size: Optional[int] = None, ngrams: Optional[int] = None
    ):
        """Trim the dictionary size to `new_max_dict_size` or self.max_dict_size.

        If `ngram` is set, apply unking after subsetting ngram vocabs up to argument.
        """

        max_dict_size = new_max_dict_size if new_max_dict_size else self.max_dict_size

        ngrams = ngrams if ngrams else self.ngram
        
        print(f"Total count: {len(self)}")

        self._remove_whitespace_tokens()
        print(f"Removed whitespaces: {len(self)}")

        min_frequency = {k: v for k, v in self.frequencies.items() if v > 10_000}
        print(f"Min occurence 10 000: {len(min_frequency)}")

        # Pre-define the number of tokens per ngram
        if ngrams <= 4:
            n_tokens_per_ngram = self._calculate_ngram_order_dict_size(ngrams, new_max_dict_size)
        else:
            n_tokens_per_ngram = list(map(lambda x: round(x*max_dict_size), utils.n_dist(ngrams, "exp")))

        print(n_tokens_per_ngram)

        # Take subset of frequencies counter based on ngram
        frequencies = []
        frequencies_tokens = []

        # For unigrams, we guarantee 200 tokens
        frequency = Counter()

        freq_dict = {}

        for token, freq in self.frequencies.items():
            if token in self.ngram2word2idx[1]:
                freq_dict[token] = freq

        frequency.update(freq_dict)
        most_common = frequency.most_common(200)

        frequencies_tokens.append(list(map(lambda x: x[0], most_common)))
        frequencies.append(most_common)

        for ngram in range(2, ngrams+1):
            frequency = Counter()

            freq_dict = {}

            for token, freq in min_frequency.items():
                if token in self.ngram2word2idx[ngram]:
                    freq_dict[token] = freq

            frequency.update(freq_dict)
            print(f"Len freq Ngram: {ngram} -> {len(frequency)}")
            most_common = frequency.most_common(n_tokens_per_ngram[ngram-1])
            print(f"Most common: {ngram} -> {len(most_common)}")

            frequencies_tokens.append(list(map(lambda x: x[0], most_common)))
            frequencies.append(most_common)

        dictionary = Dictionary(ngrams, max_dict_size, self.ngme)

        marker_tokens = ["<unk>", "<pad>", "<start>"]

        # Add all ngrams up to ngrams to new dictionary
        for ngram in self.ngram2idx2word:
            for token, ngram_idx in self.ngram2word2idx[ngram].items():
                if ngram <= ngrams:
                    if (
                        token in frequencies_tokens[ngram - 1]
                        or token in marker_tokens
                        # or ngram == 1 TODO: Do we really need to ensure all unigrams are in?
                    ):

                        if token in marker_tokens:
                            marker_idx = dictionary.add_ngram(token, ngram)
                            dictionary._marker_tokens[ngram].append(marker_idx)
                        else:
                            dictionary.add_ngram(token, ngram)

        # Collect all frequencies to pass to new dictionary
        new_frequency = Counter()
        for frequncy in frequencies:
            new_frequency.update(dict(frequncy))

        dictionary.frequencies = new_frequency

        return dictionary

    def get_marker_tokens(self) -> Iterator[int]:
        """Return an iterator over the marker tokens."""
        for token_idxs in self._marker_tokens.values():
            for token_idx in token_idxs:
                yield token_idx

    def __len__(self):
        return self.current_max_idx

    def get_item_for_index(self, idx):
        """Return the token for a given index."""
        for idxs in self.ngram2idx2word.values():
            if idx in idxs:
                return idxs[idx]
        return None

    def save_vocabulary(
        self,
        vocab_file: str,
        ngram: int,
    ) -> str:
        """Save the vocabulary to a file."""

        index = 0
        vocab = {"ngram": ngram, "ngme": self.ngme, "vocab": []}

        for ngram in range(1, self.ngram + 1):
            for idx, token in self.ngram2idx2word[ngram].items():
                if index != idx:
                    index = idx

                if "\n" in token:
                    token = token.replace("\n", "\\n")

                try:
                    frequency = self.frequencies[token]
                except KeyError:
                    frequency = -1

                index += 1
                vocab["vocab"].append({"token": token, "index": idx, "frequency": frequency, "ngram": ngram})

        with open(vocab_file, "w", encoding="utf-8") as writer:
            json.dump(vocab, writer, indent=4, ensure_ascii=False)

    def tokenize_line(
        self,
        line: Union[str, List[str]],
        id_type=torch.int64,
        return_tensor: Optional[str] = None,
        with_text: bool = True,
    ) -> dict:
        if self.ngme in ["dense", "explicit"]:
            return self._tokenize_line_explicit(line, id_type, return_tensor, with_text)
        elif self.ngme in ["sparse", "compositional"]:
            return self._tokenize_line_compositional(line, id_type, return_tensor, with_text)
        else:
            raise ValueError(f"Unknown NGME type: {self.ngme}")

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
        """Convert return value to desired format."""
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

        # Apply padding later!
        # for i in range(1, shifts + 1):
        #     st[0][-i] = self.ngram2word2idx[i]["<pad>"]
        return torch.roll(t, -shifts, 1)

    def create_weight_tensor(
        self, unigram_ppl: bool, weighted_loss: bool = True
    ) -> torch.Tensor:
        unked_freqs = self.frequencies.most_common(self.max_dict_size)

        t = torch.ones(len(self.ngram2idx2word[1])) if unigram_ppl else torch.ones(len(self))

        if not unigram_ppl and weighted_loss:
            for token, freq in unked_freqs:
                t[self.ngram2word2idx[self.token_to_n_order(token)][token]] = freq

            normed_weights = torch.tensor([(1 - (x / (max(t) + 1))).item() for x in t])
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
        """Get N-gram order for a token"""
        for n_gram, word2idx in self.ngram2word2idx.items():
            if token in word2idx:
                return n_gram

        return 0

    def print_sequence(self, seq, ngram):
        """Print a sequence of token ids"""
        collected_tokens = self._get_char_sequence(seq, ngram)
        print(f"[{', '.join(collected_tokens)}]")

    def _get_char_sequence(self, seq, ngram):
        """Convert a sequence of token ids to a list of chars"""
        collected_tokens = []

        for token in seq:
            if not isinstance(token, int):
                token = token.item()
            collected_tokens.append(self.ngram2idx2word[ngram][token])

        return collected_tokens

    def print_batch(self, batch):
        """Print a batch of token ids with dimensions (ngram, seq_len, batch)"""
        seqs = []
        for batch_idx in range(batch.size(2)):
            for n_idx in range(batch.size(0)):
                seqs.append(
                    self._get_char_sequence(batch[n_idx, :, batch_idx], n_idx + 1)
                )

        for seq in seqs:
            print(seq)

def collect_ngrams(line, n):
    """Collect ngrams from a line of text"""
    return ["".join(ngram) for ngram in nltk.ngrams(line, n)]

def add_ngrams_from_text(text: str, ngrams: List[int]):
    """Add ngrams from a text to a ngram dictionary"""
    return {ngram: collect_ngrams(text, ngram) for ngram in ngrams}

def get_unigram_tokens() -> List[str]:
    """Get unigram tokens from flair"""
    flair_device = flair.device
    flair.device = "cpu"

    # Using unigrams from flair as base
    e = FlairEmbeddings("news-forward")
    flair.device = flair_device
    return list(e.lm.dictionary.item2idx_not_encoded.keys())
