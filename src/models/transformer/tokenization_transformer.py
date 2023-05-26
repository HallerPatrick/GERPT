import os
from typing import Dict, List, Optional, Tuple, Union, Sized

from pathlib import Path

import torch

import numpy as np

from nltk import ngrams as ngram_tokenizer
from tokenizers import AddedToken
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding, EncodedInput, TruncationStrategy
from transformers.utils.generic import PaddingStrategy, TensorType, to_py_obj


from src.dictionary import load_vocab, Dictionary

from typing import TYPE_CHECKING, List, Optional, Tuple


from transformers import PreTrainedTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

class DecodedOuput:

    def __init__(self, ids: List, tokenizer):
        self.ids = ids
        self.tokenizer = tokenizer

    def decode_ngram_sequence(self, ngram: int):
        continuous_ids = []
        for i_id in range(0, len(self.ids[ngram - 1]), ngram):
            continuous_ids.append(self.ids[ngram - 1][i_id])
        return "".join(self.tokenizer.convert_ids_to_tokens(continuous_ids))

    def __repr__(self) -> str:
        return self.decode_ngram_sequence(1)

    def __str__(self) -> str:
        return self.decode_ngram_sequence(1)


class GPTNGMETokenizer(PreTrainedTokenizer):

    model_input_names = ["input_ids", "attention_mask"]
    vocab_file = "vocab.pt"
    vocab_files_names = {"vocab_file": vocab_file}

    def __init__(self, vocab_file, **kwargs):
        print(vocab_file)
        if vocab_file.endswith(".json"):
            self.dictionary = Dictionary.load_from_file(vocab_file)
        else:
            self.dictionary = torch.load(vocab_file)

        self.dictionary = self.dictionary.unking()
        # self.dictionary = self.dictionary.unking(ngrams=2, new_max_dict_size=1000, min_frequency=1000)

        super().__init__(**kwargs)

        if "\n" not in self.dictionary.ngram2word2idx[1]:
            self.dictionary.add_ngram("\n", 1)

        self.eos_token_id = self.dictionary.ngram2word2idx[1]["\n"]
        self._pad_token_id = self.dictionary.ngram2word2idx[1]["\n"]
        self.unk_token_id = self.dictionary.ngram2word2idx[1]["<unk>"]

    @property
    def pad_token_id(self):
        if isinstance(self._pad_token_id, list):
            pad_token_id = self._pad_token_id[0]
            if isinstance(pad_token_id, list):
                return pad_token_id[0]
            return pad_token_id
        return self._pad_token_id

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:

        filename = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else ""),
            self.vocab_file,
        )

        torch.save(self.dictionary, filename)

        return (filename,)

    @property
    def vocab_size(self):
        return len(self.dictionary)

    def name_or_path(self):
        return ""

    def retokenize(self, input_ids, *args, **kwargs):
        decoded = self.convert_ids_to_tokens(input_ids)
        sequence = "".join(decoded)
        new_decoded = self(sequence, *args, **kwargs)
        return new_decoded

    def _tokenize(self, text):
        ngram_sequences = []
        for n in range(1, self.dictionary.ngram + 1):

            words = ["<start>" for _ in range(1, n)]
            words.extend(list(text))

            tokens = []
            for i, word in enumerate(ngram_tokenizer(words, n)):
                if "<start>" in word:
                    word = [w for w in list(word) if w != "<start>"]
                tokens.append("".join(word))

            ngram_sequences.append(tokens)

        return ngram_sequences

    def get_idx(self, token) -> int:
        for ngram in range(1, self.dictionary.ngram + 1):
            if token in self.dictionary.ngram2word2idx[ngram]:
                return self.dictionary.ngram2word2idx[ngram][token]
        return self.dictionary.ngram2word2idx[1]["<unk>"]

    def _convert_ngram_tokens_to_ids(self, ngram_tokens: List[str]) -> List[int]:
        return [self.get_idx(token) for token in ngram_tokens]

    def convert_tokens_to_ids(self, tokens: List[List[str]]):
        if not tokens:
            return []
        return [
            self._convert_ngram_tokens_to_ids(ngram_tokens) for ngram_tokens in tokens
        ]

    def _convert_id_to_token(self, index: int) -> str:
        return self.dictionary.get_item_for_index(index)

    def __call__(self, *args, **kwargs):
        output = super().__call__(*args, **kwargs)
        # Hacky: Just return the attention mask of the first ngram sequence, they should all be the same
        # output["attention_mask"] = output["attention_mask"][0]
        return output

    def _decode(
        self, token_ids: List[List[int]], skip_special_tokens: bool = False, **kwargs
    ) -> str:
        return "".join(self.convert_ids_to_tokens(token_ids[0]))

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
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
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # encoded_inputs == one sample -> List[List[int]]

        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]
        # PHA: Check if we have a list of list of list, then we unpack
        if (
            len(required_input) != 0
            and isinstance(required_input[0], list)
            and isinstance(required_input[0][0], list)
        ):
            required_input = required_input[0]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if (
            max_length is not None
            and pad_to_multiple_of is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = (
            padding_strategy != PaddingStrategy.DO_NOT_PAD
            and len(required_input[0]) != max_length
        )
        
        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            if len(required_input) == 0:
                encoded_inputs["attention_mask"] = []
            else:
                encoded_inputs["attention_mask"] = [1] * len(required_input[0])

        if needs_to_be_padded:
            difference = max_length - len(required_input[0])

            if self.padding_side == "right":
                if return_attention_mask:

                    encoded_inputs["attention_mask"] = (
                        encoded_inputs["attention_mask"] + [0] * difference
                    )
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"]
                        + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = (
                        encoded_inputs["special_tokens_mask"] + [1] * difference
                    )
                for i in range(len(encoded_inputs[self.model_input_names[0]])):
                    encoded_inputs[self.model_input_names[0]][i] = (
                        required_input[i] + [self.pad_token_id] * difference
                    )
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [
                        0
                    ] * difference + encoded_inputs["attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [
                        self.pad_token_type_id
                    ] * difference + encoded_inputs["token_type_ids"]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [
                        1
                    ] * difference + encoded_inputs["special_tokens_mask"]

                for i in range(len(encoded_inputs[self.model_input_names[0]])):
                    encoded_inputs[self.model_input_names[0]][i] = [
                        self.pad_token_id
                    ] * difference + required_input[i]
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return encoded_inputs

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
        `self.pad_token_id` and `self.pad_token_type_id`).

        Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the
        text followed by a call to the `pad` method to get a padded encoding.

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
                `>= 7.5` (Volta).
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

        # Problem: The pad function checks if the encoded_inputs is a list or not
        # If it is a list it assumes that we have batches
        # With ngme encoding the input is always a list

        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(
            encoded_inputs[0], Mapping
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

        if required_input is None or (
            isinstance(required_input, Sized) and len(required_input) == 0
        ):
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        # PHA: First element in ngme is a list of list
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_tf_tensor(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_tensor(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    "Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

        required_input = encoded_inputs[self.model_input_names[0]]
        
        if required_input:
            if isinstance(required_input[0], (list, tuple)):
                if len(required_input[0]) > 0 and not isinstance(required_input[0][0], (list, tuple)):
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

    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens. Please Note, for *pair_ids*
        different than `None` and *truncation_strategy = longest_first* or `True`, it is not possible to return
        overflowing tokens. Such a combination of arguments will raise an error.
        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
                and `convert_tokens_to_ids` methods.
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        pair = bool(pair_ids is not None)
        if len(ids) == 0:
            len_ids = 0
        else:
            len_ids = len(ids[0])

        if pair and len(pair_ids) == 0:
            len_pair_ids = 0
        elif pair and len(pair_ids) > 0:
            len_pair_ids = len(pair_ids[0])
        else:
            len_pair_ids = 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        if (
            return_overflowing_tokens
            and truncation_strategy == TruncationStrategy.LONGEST_FIRST
            and pair_ids is not None
        ):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[List[int]], token_ids_1: Optional[List[List[int]]] = None
    ) -> List[List[int]]:
        """
        Concatenate nested ngram sequences.

        Args:
            token_ids_0 (`List[List[int]]`): The first tokenized sequence.
            token_ids_1 (`List[List[int]]`, *optional*): The second tokenized sequence.

        Returns:
            `List[List[int]]`: The model input with special tokens.
        """
        if token_ids_1 is None or len(token_ids_1) == 0:
            return token_ids_0

        if len(token_ids_0) == 0:
            return token_ids_1

        return np.concatenate((np.array(token_ids_0), np.array(token_ids_1)), axis=1).tolist()

    def truncate_sequences(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Truncates a sequence pair in-place following the strategy.
        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
                and `convert_tokens_to_ids` methods.
            num_tokens_to_remove (`int`, *optional*, defaults to 0):
                Number of tokens to remove using the truncation strategy.
            truncation_strategy (`str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
                The strategy to follow for truncation. Can be:
                - `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will truncate
                  token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a
                  batch of pairs) is provided.
                - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater
                  than the model maximum admissible input size).
            stride (`int`, *optional*, defaults to 0):
                If set to a positive number, the overflowing tokens returned will contain some tokens from the main
                sequence returned. The value of this argument defines the number of additional tokens.
        Returns:
            `Tuple[List[int], List[int], List[int]]`: The truncated `ids`, the truncated `pair_ids` and the list of
            overflowing tokens. Note: The *longest_first* strategy returns empty list of overflowing tokens if a pair
            of sequences (or a batch of pairs) is provided.
        """
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        overflowing_tokens = []
        if truncation_strategy == TruncationStrategy.ONLY_FIRST or (
            truncation_strategy == TruncationStrategy.LONGEST_FIRST and pair_ids is None
        ):

            ids = np.array(ids)

            # PHA: I think we only truncate with longest first
            if ids.shape[1] > num_tokens_to_remove:
                window_len = min(ids.shape[1], stride + num_tokens_to_remove)
                if self.truncation_side == "left":
                    overflowing_tokens = ids[:, :window_len]
                    ids = ids[:, num_tokens_to_remove:]
                elif self.truncation_side == "right":
                    overflowing_tokens = ids[-window_len:]
                    ids = ids[:, :-num_tokens_to_remove]
                else:
                    raise ValueError(f"invalid truncation strategy: {self.truncation_side}, use 'left' or 'right'.")

                ids = ids.tolist()

            else:
                error_msg = (
                    f"We need to remove {num_tokens_to_remove} to truncate the input "
                    f"but the first sequence has a length {len(ids)}. "
                )
                if truncation_strategy == TruncationStrategy.ONLY_FIRST:
                    error_msg = (
                        error_msg + "Please select another truncation strategy than "
                        f"{truncation_strategy}, for instance 'longest_first' or 'only_second'."
                    )
                logger.error(error_msg)
        elif truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            logger.warning(
                "Be aware, overflowing tokens are not returned for the setting you have chosen,"
                f" i.e. sequence pairs with the '{TruncationStrategy.LONGEST_FIRST.value}' "
                "truncation strategy. So the returned list will always be empty even if some "
                "tokens have been removed."
            )
            ids = np.array(ids)
            pair_ids = np.array(pair_ids)

            for _ in range(num_tokens_to_remove):
                if pair_ids is None or ids.shape[1] > pair_ids.shape[1]:
                    if self.truncation_side == "right":
                        ids = ids[:, :-1]
                    elif self.truncation_side == "left":
                        ids = ids[:, 1:]
                    else:
                        raise ValueError("invalid truncation strategy:" + str(self.truncation_side))
                else:
                    if self.truncation_side == "right":
                        pair_ids = pair_ids[:, :-1]
                    elif self.truncation_side == "left":
                        pair_ids = pair_ids[:, 1:]
                    else:
                        raise ValueError("invalid truncation strategy:" + str(self.truncation_side))

            ids = ids.tolist()
            pair_ids = pair_ids.tolist()

        elif truncation_strategy == TruncationStrategy.ONLY_SECOND and pair_ids is not None:
            raise NotImplementedError("PHA: I think we only truncate with longest first")
            if len(pair_ids) > num_tokens_to_remove:
                window_len = min(len(pair_ids), stride + num_tokens_to_remove)
                if self.truncation_side == "right":
                    overflowing_tokens = pair_ids[-window_len:]
                    pair_ids = pair_ids[:-num_tokens_to_remove]
                elif self.truncation_side == "left":
                    overflowing_tokens = pair_ids[:window_len]
                    pair_ids = pair_ids[num_tokens_to_remove:]
                else:
                    raise ValueError("invalid truncation strategy:" + str(self.truncation_side))
            else:
                logger.error(
                    f"We need to remove {num_tokens_to_remove} to truncate the input "
                    f"but the second sequence has a length {len(pair_ids)}. "
                    f"Please select another truncation strategy than {truncation_strategy}, "
                    "for instance 'longest_first' or 'only_first'."
                )

        return (ids, pair_ids, overflowing_tokens)
