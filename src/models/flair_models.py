"""Monkey patch flair model that loads our language models."""

from pathlib import Path
from typing import List, Optional, Union

import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import AutoModel, AutoConfig
import wrapt
from flair.data import Corpus, Sentence, Token
from flair.embeddings.token import TransformerWordEmbeddings


class NGMETransformerWordEmbeddings(TransformerWordEmbeddings):
    def __init__(
        self,
        model,
        is_document_embedding: bool = True,
        allow_long_sentences: bool = True,
        **kwargs,
    ):
        super().__init__(model, is_document_embedding, allow_long_sentences, **kwargs)

    @classmethod
    def create_from_state(cls, **state):

        del state["is_token_embedding"]

        if "vocab_file" not in state:
            state["vocab_file"] = state["model"] + "/vocab.txt"

        return cls(**state)

    def _get_begin_offset_of_tokenizer(self) -> int:
        return 0

    def _has_initial_cls_token(self) -> bool:
        return False

    def _get_processed_token_text(self, token: Token) -> str:
        pieces = self.tokenizer.tokenize(token.text)
        token_text = ""
        for piece in pieces[0]:
            token_text += self._remove_special_markup(piece)
        token_text = token_text.lower()
        return token_text

    @property
    def embedding_length(self) -> int:
        return self.model.embedding_size

    def _reconstruct_tokens_from_subtokens(self, sentence, subtokens):

        """

        sentence: Hello World

        subtokens: H e l l o  w o r l d


        """
        word_iterator = iter(sentence)
        token = next(word_iterator)
        token_text = self._get_processed_token_text(token)
        # -> Hello
        token_subtoken_lengths = []
        reconstructed_token = ""
        subtoken_count = 0
        # iterate over subtokens and reconstruct tokens

        whitespace_count = 0

        # Iter through unigram seq,
        for subtoken_id, subtoken in enumerate(subtokens[0]):

            # subtoken == char

            # remove special markup
            subtoken = self._remove_special_markup(subtoken)

            # # TODO check if this is necessary is this method is called before prepare_for_model
            # # check if reconstructed token is special begin token ([CLS] or similar)
            if subtoken in self.special_tokens and subtoken_id == 0:
                continue

            # Ignore whitespaces in subtoken (char level tokenizer)
            if subtoken == " ":
                whitespace_count += 1
                continue

            subtoken_count += 1

            # append subtoken to reconstruct token
            reconstructed_token = reconstructed_token + subtoken

            # check if reconstructed token is the same as current token
            if reconstructed_token.lower() == token_text:

                # if so, add subtoken count
                token_subtoken_lengths.append(subtoken_count)

                # reset subtoken count and reconstructed token
                reconstructed_token = ""
                subtoken_count = 0
                # break from loop if all tokens are accounted for
                if len(token_subtoken_lengths) < len(sentence):
                    token = next(word_iterator)
                    token_text = self._get_processed_token_text(token)
                else:
                    break

        # if tokens are unaccounted for
        while len(token_subtoken_lengths) < len(sentence) and len(token.text) == 1:
            token_subtoken_lengths.append(0)
            if len(token_subtoken_lengths) == len(sentence):
                break
            token = next(word_iterator)

        # check if all tokens were matched to subtokens
        if token != sentence[-1]:
            print(
                f"Tokenization MISMATCH in sentence '{sentence.to_tokenized_string()}'"
            )
            print(f"Last matched: '{token}'")
            print(f"Last sentence: '{sentence[-1]}'")
            print(f"subtokenized: '{subtokens}'")
        return token_subtoken_lengths

    def _gather_tokenized_strings(self, sentences: List[Sentence]):
        tokenized_sentences = []
        all_token_subtoken_lengths = []
        subtoken_lengths = []

        for sentence in sentences:

            # subtokenize the sentence
            tokenized_string = sentence.to_tokenized_string()

            # transformer specific tokenization
            subtokenized_sentence = self.tokenizer.tokenize(tokenized_string)

            # set zero embeddings for empty sentences and exclude
            if len(subtokenized_sentence) == 0:
                if self.token_embedding:
                    for token in sentence:
                        token.set_embedding(
                            self.name, torch.zeros(self.embedding_length)
                        )
                if self.document_embedding:
                    sentence.set_embedding(
                        self.name, torch.zeros(self.embedding_length)
                    )
                continue

            if self.token_embedding:
                all_token_subtoken_lengths.append(
                    self._reconstruct_tokens_from_subtokens(
                        sentence, subtokenized_sentence
                    )
                )

            # All ngrams seq have same length, just take unigram one
            subtoken_lengths.append(len(subtokenized_sentence[0]))

            # remember tokenized sentences and their subtokenization
            tokenized_sentences.append(tokenized_string)
        return tokenized_sentences, all_token_subtoken_lengths, subtoken_lengths

    def _add_embeddings_to_sentences(self, sentences: List[Sentence]):

        (
            tokenized_sentences,
            all_token_subtoken_lengths,
            subtoken_lengths,
        ) = self._gather_tokenized_strings(sentences)

        # encode inputs
        batch_encoding = self.tokenizer(
            tokenized_sentences,
            stride=self.stride,
            return_overflowing_tokens=self.allow_long_sentences,
            return_token_type_ids=False,
            truncation=self.truncate,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids, model_kwargs = self._build_transformer_model_inputs(
            batch_encoding, tokenized_sentences, sentences
        )

        gradient_context = (
            torch.enable_grad()
            if (self.fine_tune and self.training)
            else torch.no_grad()
        )

        # N-Gram Encoder expects ngram in first dim and batch last
        # (batch, ngram, seq) -> (ngram, seq, batch)
        input_ids = input_ids.permute((1, 2, 0))

        with gradient_context:

            hidden_states = self.model.forward_hidden(input_ids, **model_kwargs)

            # Out: (seq, batch, hid) -> (batch, seq, hid)
            hidden_states = hidden_states.permute((1, 0, 2))

            # print(hidden_states.size())

            # make the tuple a tensor; makes working with it easier.
            # (layers, batch, seq, hidden)
            # hidden_states = torch.stack(hidden_states)

            hidden_states = hidden_states.unsqueeze(0)

            # only use layers that will be outputted
            hidden_states = hidden_states[self.layer_indexes, :, :]

            if self._try_document_embedding_shortcut(hidden_states, sentences):
                return

            if self.allow_long_sentences:
                sentence_hidden_states = self._combine_strided_sentences(
                    hidden_states,
                    sentence_parts_lengths=torch.unique(
                        batch_encoding["overflow_to_sample_mapping"],
                        return_counts=True,
                        sorted=True,
                    )[1].tolist(),
                )
            else:
                sentence_hidden_states = list(hidden_states.permute((1, 0, 2, 3)))

            # remove padding tokens
            sentence_hidden_states = [
                sentence_hidden_state[:, : subtoken_length + 1, :]
                for (subtoken_length, sentence_hidden_state) in zip(
                    subtoken_lengths, sentence_hidden_states
                )
            ]

            if self.document_embedding:
                self._extract_document_embeddings(sentence_hidden_states, sentences)

            if self.token_embedding:
                self._extract_token_embeddings(
                    sentence_hidden_states, sentences, all_token_subtoken_lengths
                )


def patch_flair_lstm():

    import flair
    import torch

    from src.models.rnn import RNNModel

    @wrapt.patch_function_wrapper(flair.models.LanguageModel, "load_language_model")
    def load_language_model(wrapped, instance, args, kwargs):
        """Monkey patch load_language_model to load our RNNModel"""

        state = torch.load(str(args[0]), map_location=flair.device)

        model = RNNModel(
            dictionary=state["dictionary"],
            nlayers=state["nlayers"],
            ngrams=state["ngrams"],
            hidden_size=state["hidden_size"],
            unk_t=state["unk_t"],
            nout=state["nout"],
            embedding_size=state["embedding_size"],
            is_forward_lm=state["is_forward_lm"],
            document_delimiter=state["document_delimiter"],
            dropout=state["dropout"],
        )
        model.load_state_dict(state["state_dict"])
        model.eval()
        model.to(flair.device)

        return model

def patch_flair_trans():

    import flair
    import torch

    from src.models.rnn import RNNModel

    @wrapt.patch_function_wrapper(flair.models.LanguageModel, "load_language_model")
    def load_language_model(wrapped, instance, args, kwargs):
        """Monkey patch load_language_model to load our RNNModel"""
        
        print("LOAD TRANSFORMER")
        print(str(args[0]))
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(str(args[0]), vocab_file=str(args[0]) + "/vocab.txt", **kwargs)
        config = AutoConfig.from_pretrained(str(args[0]), output_hidden_states=True, **kwargs)
        model = AutoModel.from_pretrained(str(args[0]), config=config)
        model.is_forward_lm = True
        model.tokenizer = tokenizer
        model.to(flair.device)
        return model

def load_corpus(
    corpus_name: str, base_path: Optional[Union[str, Path]] = None
) -> Corpus:

    from flair.datasets import CONLL_03, CONLL_03_GERMAN, IMDB, UD_ENGLISH
    from flair.datasets.document_classification import (SENTEVAL_CR,
                                                        SENTEVAL_SST_BINARY)

    corpus_mapper = {
        "conll_03": {"corpus": CONLL_03, "args": [base_path]},
        "conll_03_de": {"corpus": CONLL_03_GERMAN, "args": [base_path]},
        "ud_english": {"corpus": UD_ENGLISH, "args": []},
        "imdb": {"corpus": IMDB, "args": []},
        "glue/sst2": {"corpus": SENTEVAL_SST_BINARY, "args": []},
        "senteval": {"corpus": SENTEVAL_CR, "args": []},
    }

    return corpus_mapper[corpus_name]["corpus"](*corpus_mapper[corpus_name]["args"])
