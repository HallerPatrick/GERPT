import collections
import os
import string
import sys
from typing import Dict, List, Optional, Tuple, Union

from nltk import ngrams as ngram_tokenizer
from tokenizers import AddedToken
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding, EncodedInput
from transformers.utils.generic import PaddingStrategy, TensorType

from src.dictionary import Dictionary

all_tokens = string.printable


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = iter(reader.readlines())

    try:
        ngrams = int(next(tokens).strip())
    except:
        print("Could not determine ngram of tokenizer in vocab file")

    for index, token in enumerate(tokens):
        token = token.rstrip("\n")

        if "\\n" in token:
            token = token.replace("\\n", "\n")

        vocab[token] = index

    return ngrams, vocab


class NGMETokenizerSparse(PreTrainedTokenizer):

    def __init__(self, ngrams, **kwargs):
        pad_token = AddedToken("<pad>")
        super().__init__(pad_token=pad_token, **kwargs)

        self.ngram = ngrams

        self.dictionary = Dictionary(ngrams, 0, 0, 0)

        for n_gram in range(1, ngrams + 1):
            start_idx = self.dictionary.add_ngram("<start>", n_gram)
            # pad_idx = dictionary.add_ngram("<pad>", n_gram)
            unk_idx = self.dictionary.add_ngram("<UNK>", n_gram)

            if n_gram not in self.dictionary._marker_tokens:
                self.dictionary._marker_tokens[n_gram] = [start_idx]

            for char in all_tokens:
                self.dictionary.add_ngram(char, n_gram)

        self.dictionary.set_pad_token()

    @property
    def vocab_size(self):
        return len(self.dictionary)

    def _tokenize(self, text, **kwargs):

        ngram_sequences = []
        min_length = sys.maxsize

        for n in range(1, self.ngram + 1):
            # Adding start offsets for all ngrams
            # words = ["<start>" for _ in range(1, n)]
            # words.extend(list(text))

            # s = self.shift_left(words)

            ngram_sequences.append(text)
            # ngram_target_sequences.append(s)

        return ngram_sequences

    def shift_left(self, seq):
        seq.pop(0)
        seq[-1] = self.dictionary.pad_token
        return seq

    def _convert_token_to_id(self, token, ngram, unk_token=None):
        """Converts a token (str) in an id using the vocab."""
        unk = unk_token if unk_token else self.unk_token
        try:
            return self.dictionary.ngram2word2idx[ngram][token]
        except KeyError:
            return self.dictionary.ngram2word2idx[ngram]["<unk>"]

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_ids(
            self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:

        # if tokens in [self.pad_token, self.unk_token]:
        #     return self._convert_token_to_id(tokens)
        if isinstance(tokens, int):
            return self._convert_token_to_id(tokens, 1)

        n_gram_ids = []
        for n, n_gram_seq in enumerate(tokens):
            ids = []

            for token in n_gram_seq:
                ids.append(self._convert_token_to_id(token, n + 1, f"<{n + 1}-UNK>"))

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
            if is_tf_available() and _is_tensorflow(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_available() and _is_torch(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    f"Should be one of a python, numpy, pytorch or tensorflow object."
                )

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
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if len(encoded_inputs[self.model_input_names[0]]) == 0:
            return encoded_inputs

        required_input = encoded_inputs[self.model_input_names[0]]
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

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input_first)

        if needs_to_be_padded:
            difference = max_length - len(required_input_first)

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
                for n, n_seq in enumerate(encoded_inputs[self.model_input_names[0]]):
                    encoded_inputs[self.model_input_names[0]][n] = (
                            n_seq + [self.pad_token_id] * difference
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
                for n, n_seq in enumerate(encoded_inputs[self.model_input_names[0]]):
                    encoded_inputs[self.model_input_names[0]][n] = [
                                                                       self.pad_token_id
                                                                   ] * difference + n_seq
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return encoded_inputs


class NGMETokenizer(PreTrainedTokenizer):
    vocab_file_name = "vocab.txt"
    eod = None

    def __init__(
            self,
            vocab_file: Optional[str] = None,
            unk_token="<unk>",
            pad_token="<pad>",
            **kwargs,
    ):

        unk_token = AddedToken(unk_token) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token) if isinstance(pad_token, str) else pad_token

        if vocab_file:
            self.ngrams, self.vocab = load_vocab(vocab_file)
        else:
            assert "name_or_path" in kwargs
            self.vocab = load_vocab(kwargs["name_or_path"] + "/" + self.vocab_file_name)

        self.decoder = {v: k for k, v in self.vocab.items()}
        super().__init__(pad_token=pad_token, unk_token=unk_token, **kwargs)

        # self.add_special_tokens({"pad_token": "<pad>"})

        # self.pad_token_id = 0

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

            ngram_sequences.append(tokens)
        return ngram_sequences

    def _convert_token_to_id(self, token, unk_token=None):
        """Converts a token (str) in an id using the vocab."""
        unk = unk_token if unk_token else self.unk_token
        return self.vocab.get(token, self.vocab.get(unk))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_ids(
            self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:

        # if tokens in [self.pad_token, self.unk_token]:
        #     return self._convert_token_to_id(tokens)

        if isinstance(tokens, int) or isinstance(tokens, str):
            return self._convert_token_to_id(tokens, 1)

        n_gram_ids = []
        for n, n_gram_seq in enumerate(tokens):
            ids = []

            for token in n_gram_seq:
                ids.append(self._convert_token_to_id(token, "<unk>"))

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
            if is_tf_available() and _is_tensorflow(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_available() and _is_torch(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    f"Should be one of a python, numpy, pytorch or tensorflow object."
                )

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
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]
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

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input_first)

        if needs_to_be_padded:
            difference = max_length - len(required_input_first)

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
                for n, n_seq in enumerate(encoded_inputs[self.model_input_names[0]]):
                    encoded_inputs[self.model_input_names[0]][n] = (
                            n_seq + [self.pad_token_id] * difference
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
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "")
                + self.vocab_file_name,
            )
        else:
            vocab_file = (
                             filename_prefix + "-" if filename_prefix else ""
                         ) + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            writer.write(str(self.ngrams) + "\n")
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    print(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index

                if "\n" in token:
                    token = token.replace("\n", "\\n")
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
