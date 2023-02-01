from functools import lru_cache
import itertools
import os
import sys
import copy

import nltk
from pathlib import Path
from typing import Optional, List, Union, Iterator
from collections import defaultdict, Counter


import torch

from src import utils

# from src.utils import display_input_n_gram_sequences, display_text

class Dictionary:
    def __init__(
            self, ngram: int, max_dict_size: int, unk_threshold: int, fallback: bool, ngme: str, packed: bool = False
    ):
        self._marker_tokens = {}
        self.ngram_indexes = defaultdict(list)
        self.ngram = ngram
        self.max_dict_size = max_dict_size
        self.unk_threshold = unk_threshold
        self.fallback = fallback
        self.frequencies: Counter = Counter()
        self.pad_tokens = {}

        self.ngram2word2idx = {}
        self.ngram2idx2word = {}

        self.current_max_idx = 0
        self.ngme = ngme
        self.packed = packed

    def get_all_tokens(self):
        for ngram in range(1, self.ngram+1):
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

    def unking(self):

        candidates = list(
            map(
                lambda x: x[0], self.frequencies.most_common(self.max_dict_size)
            )
        )

        dictionary = Dictionary(self.ngram, self.max_dict_size, self.unk_threshold, self.fallback, self.ngme)

        for ngram in self.ngram2idx2word.keys():
            for token, ngram_idx in self.ngram2word2idx[ngram].items():
                if token in candidates or ngram == 1:
                    dictionary.add_ngram(token, ngram)
        
        dictionary.frequencies = self.frequencies

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
            writer.write(str(ngram) + " " + self.ngme + "\n")
            
            for ngram in range(1, self.ngram+1):
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

    def tokenize_line(self, line: Union[str, List[str]], id_type=torch.int16) -> dict:
        if self.ngme == "dense":
            return self._tokenize_line_dense(line, id_type)
        elif self.ngme == "sparse":
            return self._tokenize_line_sparse(line, id_type)
        else:
            raise ValueError("UNKOWN NGME APPROACH")
    
    @lru_cache
    def _special_tokens(self) -> List[str]:
        return [self.ngram2idx2word[1][special_token_id] for special_token_id in self._marker_tokens[1]]


    def _tokenize_line_dense(self, line: Union[str, List[str]], id_type):
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

            seq = torch.tensor(ids).type(torch.int64).unsqueeze(dim=0)

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


        return {"text": line, "source": sequence.numpy(), "target": target.numpy()}

    def _tokenize_line_sparse(self, line: Union[str, List[str]], id_type):
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
        

        return {"source": sequence, "target": target}

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

    def create_weight_tensor(self, unigram_ppl: bool, weighted_loss: bool = True) -> torch.Tensor:

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
            if marker < len(normed_weights) and self.get_item_for_index(marker) != "<eod>":
                normed_weights[marker] = 0

        return normed_weights

    def token_to_n_order(self, token: str) -> int:

        for n in self.ngram2word2idx:
            if token in self.ngram2word2idx[n]:
                return n

        return 0

    def print_sequence(self, seq, ngram):
        """docstring for print_sequence"""
        
        collected_tokens = []

        for token in seq:
            if not isinstance(token, int):
                token = token.item()
            collected_tokens.append(self.ngram2idx2word[ngram][token])

        print(f"[{', '.join(collected_tokens)}]")
