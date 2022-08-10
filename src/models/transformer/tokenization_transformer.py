import os
import collections
import json
from typing import Optional, Tuple, Union, List

from nltk import ngrams as ngram_tokenizer
from tokenizers import AddedToken

from transformers import PreTrainedTokenizer
from transformers import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig

from src.dataset import Dictionary

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()

    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        
        if "\\n" in token:
            token = token.replace("\\n", "\n")

        vocab[token] = index

    return vocab

class NGMETokenizer(PreTrainedTokenizer):

    vocab_file_name = "vocab.json"

    def __init__(self, vocab_file: str, unk_token="<1-UNK>", pad_token="<pad>", **kwargs):

        unk_token = AddedToken(unk_token) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token) if isinstance(pad_token, str) else pad_token


        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            **kwargs
        )
        
        # Not sure if this always works
        try:
            config = AutoConfig.from_pretrained(kwargs["name_or_path"])
            self.ngrams = config.ngrams
        except KeyError:
            self.ngrams = 1
        # self.fallback = config.fallback

        self.vocab = load_vocab(vocab_file)

        self.decoder = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text, **kwargs):

        ngram_sequences = []

        for n in range(1, self.ngrams + 1):
            words = ["<start>" for _ in range(1, n)]
            words.extend(list(text))

            tokens = []

            for _, word in enumerate(ngram_tokenizer(words, n)):

                if "<start>" in word:
                    word = [w for w in list(word) if w != "<start>"]

                tokens.append("".join(word))
        
                # try:
                #     ids.append(self.vocab["".join(word)])
                # except KeyError:
                #
                #     # Fall back on n-1 gram if possible
                #     # if self.fallback and tuple(word)[1:] in self.vocab:
                #     #     ids.append(self.vocab[word])
                #     # else:
                #     ids.append(self.vocab[f"<{n}-UNK>"])
                # length += 1

            ngram_sequences.append(tokens)
        
        return ngram_sequences

    def _convert_token_to_id(self, token, unk_token=None):
        """Converts a token (str) in an id using the vocab."""
        unk = unk_token if unk_token else self.unk_token
        return self.vocab.get(token, self.vocab.get(unk))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:

        if tokens in [self.pad_token, self.unk_token]:
            return self._convert_token_to_id(tokens)
        
    
        n_gram_ids = []
        for n, n_gram_seq in enumerate(tokens):
            ids = []

            for token in n_gram_seq:
                ids.append(self._convert_token_to_id(token, f"<{n}-UNK>"))
            
            n_gram_ids.append(ids)

        return n_gram_ids

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + self.vocab_file_name
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    print(index, token_index)
                    print(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return ( vocab_file, )
