import os
import string
import sys
from typing import Dict, List, Optional, Tuple, Union

from pathlib import Path

from nltk import ngrams as ngram_tokenizer
from tokenizers import AddedToken
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding, EncodedInput
from transformers.utils.generic import PaddingStrategy, TensorType, to_py_obj


from src.dictionary import load_vocab


class NGMETokenizer(PreTrainedTokenizer):
    vocab_file_name = "vocab.txt"
    eod = None

    model_input_names = ["input_ids"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        unk_token="<unk>",
        pad_token="<pad>",
        **kwargs,
    ):

        unk_token = AddedToken(unk_token) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token) if isinstance(pad_token, str) else pad_token

        self.ngrams, self.ngme_type, self.vocab = load_vocab(vocab_file)

        self.token_to_ngram = {}
        self.idx_to_ngram = {}

        self.decoder = {}

        for ngram in self.vocab:

            for token in self.vocab[ngram]:
                self.token_to_ngram[token] = ngram
                self.idx_to_ngram[self.vocab[ngram][token]] = ngram

                if ngram not in self.decoder:
                    self.decoder[ngram] = {self.vocab[ngram][token]: token}
                else:
                    self.decoder[ngram][self.vocab[ngram][token]] = token

        # if "<eod>" in self.token_to_ngram:
        #     self.eod_id = self.vocab[1]["<eod>"]

        super().__init__(pad_token=pad_token, unk_token=unk_token, **kwargs)

        # self.add_special_tokens({"pad_token": "<pad>"})

        # self.pad_token_id = 0

    @property
    def eod_id(self):
        print(self.vocab[1])
        if "<eod>" in self.vocab[1]:
            return self.vocab[1]["<eod>"]

    @property
    def vocab_size(self):
        return sum([len(sub) for sub in self.vocab.values()])

    def _tokenize(self, text, **kwargs):

        if self.ngme_type == "dense":
            return self._tokenize_dense(text, **kwargs)

        elif self.ngme_type == "sparse":
            return self._tokenize_sparse(text, **kwargs)
        else:
            raise ValueError(f"NGME approach: {self.ngme_type} is unknown")

    def _tokenize_dense(self, text, **kwargs):
        ngram_sequences = []

        for n in range(1, self.ngrams + 1):
            words = ["<start>" for _ in range(1, n)]
            words.extend(list(text))

            if "check_special_tokens" in kwargs and kwargs["check_special_tokens"]:

                if isinstance(text, list):
                    text = "".join(text)

                text = text.replace("<eod>", "Ġ")
                words.extend(list(text))
                words[words.index("Ġ")] = "<eod>"
            else:
                words.extend(list(text))

            tokens = []

            for _, word in enumerate(ngram_tokenizer(words, n)):

                if "<start>" in word:
                    word = [w for w in list(word) if w != "<start>"]

                tokens.append("".join(word))

            ngram_sequences.append(tokens)
        return ngram_sequences

    def _tokenize_sparse(self, text, **kwargs):
        ngram_sequences = []

        for n in range(1, self.ngrams + 1):
            words = ["<start>" for _ in range(1, n)]

            if "check_special_tokens" in kwargs and kwargs["check_special_tokens"]:

                if isinstance(text, list):
                    text = "".join(text)

                text = text.replace("<eod>", "Ġ")
                words.extend(list(text))
                words[words.index("Ġ")] = "<eod>"
            else:
                words.extend(list(text))

            words = words[: len(text)]
            ngram_sequences.append(words)

        return ngram_sequences

    def get_token_ngram_order(self, token):
        if token in self.token_to_ngram:
            return self.token_to_ngram[token]

        return -1

    def _convert_token_to_id(self, token, ngram, unk_token="<unk>"):
        """Converts a token (str) in an id using the vocab."""
        unk = unk_token if unk_token else self.unk_token
        return self.vocab[ngram].get(token, self.vocab[ngram].get(unk))

    def _convert_id_to_token(self, index, ngram):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder[ngram].get(index)

    def _convert_ids_to_tokens(self, ids: List[int], ngram: int):
        return [self._convert_id_to_token(idx, ngram) for idx in ids]

    def convert_ids_to_tokens(self, ids: List[List[int]]):
        return [self._convert_ids_to_tokens(idxs, n + 1) for n, idxs in enumerate(ids)]

    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:

        if isinstance(tokens, int) or isinstance(tokens, str):
            # Check if we can find ngram
            ngram = self.get_token_ngram_order(tokens)
            if ngram > 0:
                return self._convert_token_to_id(tokens, ngram)
            else:
                return self._convert_token_to_id(tokens, 1)

        n_gram_ids = []
        for n, n_gram_seq in enumerate(tokens):
            ids = []

            for token in n_gram_seq:
                ids.append(self._convert_token_to_id(token, n + 1))

            n_gram_ids.append(ids)

        return n_gram_ids

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.

        Padding side (left/right) padding token ids are defined at the tokenizer level (with `self.padding_side`,
        `self.pad_token_id` and `self.pad_token_type_id`)

        <Tip>

        If the `encoded_inputs` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
        result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
        PyTorch tensors, you will lose the specific device of your tensors however.

        </Tip>

        Args:
            encoded_inputs ([`BatchEncoding`], list of [`BatchEncoding`], `Dict[str, List[int]]`, `Dict[str, List[List[int]]` or `List[Dict[str, List[int]]]`):
                Tokenized inputs. Can represent one input ([`BatchEncoding`] or `Dict[str, List[int]]`) or a batch of
                tokenized inputs (list of [`BatchEncoding`], *Dict[str, List[List[int]]]* or *List[Dict[str,
                List[int]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
                collate function.

                Instead of `List[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors), see
                the note above for the return type.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(
            encoded_inputs[0], (dict, BatchEncoding)
        ):
            encoded_inputs = {
                key: [example[key] for example in encoded_inputs]
                for key in encoded_inputs[0].keys()
            }

        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if not required_input:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            return_tensors = "pt" if return_tensors is None else return_tensors

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs[0]) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = ...,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:

        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """

        required_input_first = encoded_inputs[self.model_input_names[0]][0]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input_first)

        if (
            max_length is not None
            and pad_to_multiple_of is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = (
            padding_strategy != PaddingStrategy.DO_NOT_PAD
            and len(required_input_first) != max_length
        )

        if needs_to_be_padded:
            difference = max_length - len(required_input_first)

            if self.padding_side == "right":
                for n, n_seq in enumerate(encoded_inputs[self.model_input_names[0]]):
                    encoded_inputs[self.model_input_names[0]][n] = (
                        n_seq + [self.pad_token_id] * difference
                    )
            elif self.padding_side == "left":
                for n, n_seq in enumerate(encoded_inputs[self.model_input_names[0]]):
                    encoded_inputs[self.model_input_names[0]][n] = [
                        self.pad_token_id
                    ] * difference + n_seq
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return encoded_inputs

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:

        if isinstance(save_directory, Path):
            save_directory = str(save_directory)

        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + "vocab.txt",
            )
        else:
            vocab_file = (
                filename_prefix + "-" if filename_prefix else ""
            ) + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            writer.write(str(self.ngrams) + " " + self.ngme_type + "\n")

            for ngram in range(1, self.ngrams + 1):
                for idx, token in self.decoder[ngram].items():
                    if index != idx:
                        print(
                            f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                            " Please check that the vocabulary is not corrupted!"
                        )
                        index = idx

                    if "\n" in token:
                        token = token.replace("\n", "\\n")

                    writer.write(str(ngram) + " " + token + "\n")
                    index += 1
        return (vocab_file,)

# coding=utf-8
# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for GPTNeoX."""
import json
from typing import TYPE_CHECKING, List, Optional, Tuple

from tokenizers import pre_tokenizers

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import logging


if TYPE_CHECKING:
    from transformers.pipelines.conversational import Conversation


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}


class GPTNeoXTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" GPT-NeoX-20B tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import GPTNeoXTokenizerFast

    >>> tokenizer = GPTNeoXTokenizerFast.from_pretrained("gpt2")
    >>> tokenizer("Hello world")["input_ids"]
    [15496, 995]

    >>> tokenizer(" Hello world")["input_ids"]
    [18435, 995]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer, but since
    the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The end of sequence token.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (GPTNeoX tokenizer detect beginning of words by the preceding space).
        trim_offsets (`bool`, *optional*, defaults to `True`):
            Whether or not the post-processing step should trim offsets to avoid including whitespaces.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        self.add_prefix_space = add_prefix_space

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    def _build_conversation_input_ids(self, conversation: "Conversation") -> List[int]:
        """This corresponds to DialoGPT variants of models."""
        input_ids = []
        for is_user, text in conversation.iter_texts():
            input_ids.extend(self.encode(text, add_special_tokens=False) + [self.eos_token_id])

        if len(input_ids) > self.model_max_length:
            input_ids = input_ids[-self.model_max_length :]
        return input_ids

