import sys
from typing import Dict, List

import torch
import numpy as np
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
    "nums": {
        "train": "data/nums/train.txt",
        "test": "data/nums/test.txt",
        "validation": "data/nums/valid.txt",
    },
    "wikitext-2": {
        "train": "data/wikitext-2/train.txt",
        "test": "data/wikitext-2/test.txt",
        "validation": "data/wikitext-2/valid.txt",
    },
    # HC
    "cc100_german": {
        "train": "home/tmp/halerpat/data/train.txt",
        "test": "home/tmp/halerpat/data/test.txt",
        "validation": "home/tmp/halerpat/data/valid.txt",
    },
    "wikipedia_en": {
        "args": ["wikipedia", "20220301.en"]
    },
    "wikipedia_de": {
        "args": ["wikipedia", "20220301.de"]
    },
    "obw_news": {
        "train": "data/obw_news/train.txt"
    }
}



def tokenize_batch(
    dictionary, lines: List[str], ngram: int, label=None, otf=False, fallback=False
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

            ids = []
            for c in words:
                try:
                    ids.append(dictionary.ngram2word2idx[n][c])
                except KeyError:
                    ids.append(dictionary.ngram2word2idx[n]["<unk>"])


            if len(ids) > len_longest_chunk:
                len_longest_chunk = len(ids)

            idss_n.append(ids)

        n_gram_sequences.append(idss_n)

    padded_char_sequence = []
    for n, ls in enumerate(n_gram_sequences):
        new_lines = []
        for line in ls:
            line += [dictionary.ngram2word2idx[n+1][" " * ( n+1 )]] * (len_longest_chunk - len(line))
            new_lines.append(torch.tensor(line).type(torch.int64))

        seq = torch.cat(new_lines).unsqueeze(dim=0)

        padded_char_sequence.append(seq)

    n_gram_sequences = torch.cat([t.clone().detach() for t in padded_char_sequence])
    return n_gram_sequences

def grouped(iterable, n):
    # s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ...
    return zip(*[iter(iterable)] * n)

def batchify(text_list: str, batch_size: int, bptt: int):
    """Splits text into batches of size batch_size and bptt.
    Parameters
    ----------
    text: str
        Text to be batchified
    batch_size: int
        Number of batches
    bptt: int
        Number of tokens per batch
    """
    tokens = np.concatenate([np.array(sublist) for sublist in text_list if len(sublist) != 0], axis=1)

    text: np.ndarray = tokens

    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    nbatch = text.shape[1] // (batch_size * ( bptt + 1 ))
    text = text[:, : nbatch * batch_size * ( bptt + 1)]


    # [ngram, batched_seq, batch]
    text = text.reshape((text.shape[0], batch_size, -1)).transpose((0, 2, 1))
    
    print(text.shape)
    exit()

    new_text = []
    
    # Join all batch sequences together with backpropagation trough time
    for i in range(0, nbatch+1):

        seq = text[:, i * bptt : (i + 1) * bptt + 1] #.reshape(text.shape[0], -1, order="F")
        print(seq.shape)

        if seq.shape[1] > bptt :
            new_text.append(seq)
    
    return torch.tensor(np.array(new_text))


def prepare_data(text, batch_size, bptt_size):

    #[ngram, sequence]
    tokens = np.concatenate([np.array(sublist) for sublist in text if len(sublist) != 0], axis=1)
    tokens = torch.tensor(tokens)

    nbatch = tokens.size(1) // batch_size

    data = tokens.narrow(1, 0, nbatch * batch_size)

    data = data.view(tokens.size(0), batch_size, -1).transpose(1, 2).contiguous()
    
    return data
    # 
    # ngram = tokens.shape[0]
    #
    # # Get the number of tokens in the text
    # num_tokens = tokens.shape[1]
    #
    # # Calculate the number of batches
    # print(num_tokens, batch_size * bptt_size)
    # num_batches = num_tokens // (batch_size * bptt_size)
    #
    # # Truncate the tokens to the nearest multiple of the batch size and bptt size
    # tokens = tokens[:, :num_batches * batch_size * bptt_size]
    #
    # # Reshape the tokens into a 2D array of shape (batch_size, num_batches * bptt_size)
    # tokens_batch = np.reshape(tokens, (ngram, batch_size, num_batches * bptt_size))
    #
    # # Split the 2D array into a list of batches
    # batches = np.split(tokens_batch, num_batches, axis=2)
    # 
    # return batches
    # # For each batch, create input and target sequences
    # for i in range(num_batches):
    #     input_seq = batches[i][:, :-1]
    #     target_seq = batches[i][:, 1:]
    #     
    #     yield input_seq, target_seq
