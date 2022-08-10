"""Monkey patch flair model that loads our language models."""

from pathlib import Path
from typing import Optional, Union, List
from flair.embeddings.token import TransformerWordEmbeddings

import wrapt
from flair.data import Corpus, Sentence
import torch




class NGMETransformerWordEmbeddings(TransformerWordEmbeddings):
    

    def __init__(self, model: str = "bert-base-uncased", is_document_embedding: bool = False, allow_long_sentences: bool = True, **kwargs):
        super().__init__(model, is_document_embedding, allow_long_sentences, **kwargs)
    
    def _get_begin_offset_of_tokenizer(self) -> int:
        return 0

    def _has_initial_cls_token(self) -> bool:
        return False

    def _gather_tokenized_strings(self, sentences: List[Sentence]):
        tokenized_sentences = []
        for sentence in sentences:

            # subtokenize the sentence
            tokenized_string = sentence.to_tokenized_string()

            # transformer specific tokenization
            subtokenized_sentence = self.tokenizer.tokenize(tokenized_string)

            # set zero embeddings for empty sentences and exclude
            if len(subtokenized_sentence) == 0:
                if self.token_embedding:
                    for token in sentence:
                        token.set_embedding(self.name, torch.zeros(self.embedding_length))
                if self.document_embedding:
                    sentence.set_embedding(self.name, torch.zeros(self.embedding_length))
                continue

            # remember tokenized sentences and their subtokenization
            tokenized_sentences.append(tokenized_string)
        return tokenized_sentences

    def _add_embeddings_to_sentences(self, sentences: List[Sentence]):

        print(sentences)
        tokenized_sentences = self._gather_tokenized_strings(sentences)

        # encode inputs
        batch_encoding = self.tokenizer(
            tokenized_sentences,
            stride=self.stride,
            return_overflowing_tokens=self.allow_long_sentences,
            truncation=self.truncate,
            padding=True,
            return_tensors="pt",
        )

        input_ids, model_kwargs = self._build_transformer_model_inputs(batch_encoding, tokenized_sentences, sentences)

        gradient_context = torch.enable_grad() if (self.fine_tune and self.training) else torch.no_grad()
    
        print(input_ids)
        with gradient_context:
            hidden_states = self.model(input_ids, **model_kwargs)

            print(hidden_states)
            hidden_states = hidden_states[-1]
            print(hidden_states.size())

            # make the tuple a tensor; makes working with it easier.
            hidden_states = torch.stack(hidden_states)

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
            # sentence_hidden_states = [
            #     sentence_hidden_state[:, : subtoken_length + 1, :]
            #     for (subtoken_length, sentence_hidden_state) in zip(subtoken_lengths, sentence_hidden_states)
            # ]

            if self.document_embedding:
                self._extract_document_embeddings(sentence_hidden_states, sentences)

            # if self.token_embedding:
            #     self._extract_token_embeddings(sentence_hidden_states, sentences, all_token_subtoken_lengths)


def patch_flair():

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
            gen_args={},
        )
        model.load_state_dict(state["state_dict"])
        model.eval()
        model.to(flair.device)

        return model


def load_corpus(
    corpus_name: str, base_path: Optional[Union[str, Path]] = None
) -> Corpus:

    from flair.datasets import CONLL_03, UD_ENGLISH, IMDB, CONLL_03_GERMAN
    from flair.datasets.document_classification import SENTEVAL_SST_BINARY

    corpus_mapper = {
        "conll_03": {"corpus": CONLL_03, "args": [base_path]},
        "conll_03_de": {"corpus": CONLL_03_GERMAN, "args": [base_path]},
        "ud_english": {"corpus": UD_ENGLISH, "args": []},
        "imdb": {"corpus": IMDB, "args": []},
        "glue/sst2": {"corpus": SENTEVAL_SST_BINARY, "args": []},
    }

    return corpus_mapper[corpus_name]["corpus"](*corpus_mapper[corpus_name]["args"])
