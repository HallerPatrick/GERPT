"""Monkey patch flair model that loads our language models."""

import sys
from pathlib import Path
from typing import Optional, Union

import wrapt
from flair.data import Corpus


def patch_flair(model_name: str):

    import flair
    import torch

    from src.models.rnn import RNNModel
    from src.models.transformer import TransformerModel

    @wrapt.patch_function_wrapper(flair.models.LanguageModel, "load_language_model")
    def load_language_model(wrapped, instance, args, kwargs):
        """Monkey patch load_language_model to load our RNNModel"""

        state = torch.load(str(args[0]), map_location=flair.device)

        if model_name == "rnn":
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
        else:
            model = TransformerModel(
                dictionary=state["dictionary"],
                nlayers=state["nlayers"],
                nhead=state["nhead"],
                ngrams=state["ngrams"],
                nhid=state["hidden_size"],
                unk_t=state["unk_t"],
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
