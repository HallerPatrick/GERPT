import os
import sys

from pathlib import Path
from typing import Optional, List, Union
from collections import defaultdict, Counter


import torch

# from src.utils import display_input_n_gram_sequences, display_text

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
        self._pad_token = None

        self.ngram2word2idx = {}
        self.ngram2idx2word = {}

        self.current_max_idx = 0

    @property
    def pad_token(self):
        if not self._pad_token:
            raise ValueError("No pad token set")
        return self._pad_token

    def set_pad_token(self, pad_token="<pad>"):
        self._pad_token = self.current_max_idx
        self.current_max_idx += 1


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
            self.current_max_idx += 1
        else:
            if word not in self.ngram2word2idx[ngram]:
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

    def tokenize_line(self, line: List[str], id_type=torch.int16) -> dict:
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
        
        # print(line)


        return {"source": sequence, "target": target}

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

    # def _shift_left(self, t: torch.Tensor, ngram) -> torch.Tensor:
    #     st = torch.roll(t, -1, 1)
    #
    #     st[0][-1] = self.ngram2word2idx[ngram]["<pad>"]
    #     # for i in range(1, shifts + 1):
    #     #     st[0][-i] = self.ngram2word2idx[i]["<pad>"]
    #     return st

    def create_weight_tensor(self) -> Optional[list]:
        # if not self.frequencies:
        #     return

        t = torch.ones(len(self))

        normed_weights = t

        for marker in self.get_marker_tokens():
            normed_weights[marker] = 0

        return normed_weights

