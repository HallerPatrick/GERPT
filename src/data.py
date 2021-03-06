import sys
from collections import Counter
from typing import Dict, List, Optional

import torch
from colorama import Fore
from nltk import ngrams
from rich.progress import track

local_dataset_mapper = {
    "hp": {
        "train": "data/hp/train.txt",
        "test": "data/hp/test.txt",
        "validation": "data/hp/valid.txt",
    },
    "cash": {
        "train": "data/cash/train.txt",
        "test": "data/cash/test.txt",
        "validation": "data/cash/valid.txt",
    },
    "wikitext-2": {
        "train": "data/wikitext-2/train.txt",
        "test": "data/wikitext-2/test.txt",
        "validation": "data/wikitext-2/valid.txt",
    }
}


def tokenize_batch(
    dictionary, lines: List[str], ngram, label=None, otf=False, fallback=False
):
    """Tokenizes lines of text. Number of lines is already number of batches.
    Parameters
    ----------
    lines: List[str]
        List of strings, every string can represent a sentence or line of text.
    otf: bool
        On the Fly (oft) tokenization that leaves out the <eos> marker token,
        used for text generating of not complete sentence
    fallback: bool
        If if n-gram token is UNK try using the n-1-gram token
    """

    n_gram_sequences = []

    padding_char_index = dictionary.get_idx_for_item(" ")

    len_longest_chunk: int = 0

    for n in range(1, ngram + 1):
        idss_n = []

        _lines = (
            track(lines, description=f"Tokenize for {n}-gram sequence for {label}")
            if label
            else lines
        )
        for line in _lines:

            # Adding start offsets for all ngrams
            words = ["<start>" for _ in range(1, n)]
            words.extend(list(line))
            if not otf:
                words.append("<eos>")

            ids = []
            for word in ngrams(words, n):
                word = "".join(word)
                try:
                    ids.append(dictionary.word2idx[word])
                except KeyError:
                    # Fall back on n-1 gram if possible
                    if fallback and word[1:] in dictionary.word2idx:
                        ids.append(dictionary.word2idx[word])
                    else:
                        ids.append(dictionary.word2idx[f"<{n}-UNK>"])

            if len(ids) > len_longest_chunk:
                len_longest_chunk = len(ids)

            idss_n.append(ids)

        n_gram_sequences.append(idss_n)

    padded_char_sequence = []
    for ls in n_gram_sequences:
        new_lines = []
        for line in ls:
            line += [padding_char_index] * (len_longest_chunk - len(line))
            new_lines.append(torch.tensor(line).type(torch.int64))

        seq = torch.cat(new_lines).unsqueeze(dim=0)

        padded_char_sequence.append(seq)

    n_gram_sequences = torch.cat([torch.tensor(t) for t in padded_char_sequence])
    return n_gram_sequences


def tokenize(dictionary, lines: List[str], ngram, label, otf=False, fallback=False):
    """Tokenizes lines of text.

    Parameters
    ----------

    lines: List[str]
        List of strings, every string can represent a sentence or line of text.
    otf: bool
        On the Fly (oft) tokenization that leaves out the <eos> marker token,
        used for text generating of not complete sentence
    fallback: bool
        If if n-gram token is UNK try using the n-1-gram token
    """

    n_gram_sequences = []
    min_length = sys.maxsize

    for n in range(1, ngram + 1):
        idss_n = []

        _lines = (
            track(
                lines,
                description=f"Tokenize for {n}-gram sequence for {Fore.GREEN}{label}{Fore.RESET}",
            )
            if label
            else lines
        )

        for line in _lines:

            # Adding start offsets for all ngrams
            words = ["<start>" for _ in range(1, n)]
            words.extend(list(line))
            if not otf:
                words.append("<eos>")

            ids = []
            length = 0
            for i, word in enumerate(ngrams(words, n)):
                try:
                    ids.append(dictionary.word2idx["".join(word)])
                except KeyError:
                    # Fall back on n-1 gram if possible
                    if fallback and word[1:] in dictionary.word2idx:
                        ids.append(dictionary.word2idx[word])
                    else:
                        ids.append(dictionary.word2idx[f"<{n}-UNK>"])
                length += 1

            idss_n.append(torch.tensor(ids).type(torch.int64))

        # N-gram sequence, [1, #tokens]
        seq = torch.cat(idss_n).unsqueeze(dim=0)
        length = seq.size(1)

        if length < min_length:
            min_length = length

        n_gram_sequences.append(seq)

    n_gram_sequences = torch.cat([t[:min_length] for t in n_gram_sequences])

    return n_gram_sequences


def grouped(iterable, n):
    # s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ...
    return zip(*[iter(iterable)] * n)


# TODO: Make a hugginface dataloader here
def prep_enwiki8():
    # From: https://github.com/salesforce/awd-lstm-lm/blob/master/data/enwik8/prep_enwik8.py

    import os
    import sys
    import zipfile

    import requests

    if os.path.exists("data/enwik8/train.txt"):
        print("Tokenized enwik8 already exists - skipping processing")
        sys.exit()

    try:
        data = zipfile.ZipFile("enwik8.zip").read("enwik8")
    except:
        r = requests.get("https://data.deepai.org/enwik8.zip", stream=True)

        with open("enwik8.zip", "wb") as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)

        data = zipfile.ZipFile("enwik8.zip").read("enwik8")

    print("Length of enwik8: {}".format(len(data)))

    num_test_chars = 5000000

    train_data = data[: -2 * num_test_chars]
    valid_data = data[-2 * num_test_chars : -num_test_chars]
    test_data = data[-num_test_chars:]

    os.mkdir("data/enwik8")

    for fn, part in [
        ("data/enwik8/train.txt", train_data),
        ("data/enwik8/valid.txt", valid_data),
        ("data/enwik8/test.txt", test_data),
    ]:
        print("{} will have {} bytes".format(fn, len(part)))
        print("- Tokenizing...")
        part_str = " ".join([str(c) if c != ord("\n") else "\n" for c in part])
        print("- Writing...")
        f = open(fn, "w").write(part_str)
        f = open(fn + ".raw", "wb").write(part)
