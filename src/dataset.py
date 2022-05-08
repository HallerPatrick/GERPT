
from nltk import ngrams as ngram_tokenizer
from collections import Counter
from typing import Union
from colorama.ansi import Fore
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from src.data import Dictionary


def load_dictionary_from_path(dataset_dir: str) -> Dictionary:
    """Load the dictionary, either through a saved dictionary or 
    by building a new one from data."""

    frequences = Counter()

def load_dictionary_from_hf(source: Dataset, ngrams: int, max_dict_size: int, unk_threshold: int):

    dictionary = Dictionary()
    frequencies = Counter()

    for train_split in ["train", "test", "valid"]:
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
    return dictionary


def remove_marker_tokens(token, dictionary):
    """Due to some str length comparison to determine what n-gram the token
    is. We replace the marker tokens, with a single char, for easy comparison
    """
    for marker in dictionary.get_marker_tokens():
        token = token.replace(marker, "i")

    return token


