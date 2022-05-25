from argparse import Namespace
from typing import Dict

from src.dataset import Dictionary
from src.models.rnn import RNNModel
from src.models.transformer import TransformerModel


def load_model(dictionary: Dictionary, args: Namespace, gen_args: Dict):

    if args.model == "rnn":
        model = RNNModel(
            dictionary,
            args.nlayers,
            args.ngram,
            args.hidden_size,
            args.unk_threshold,
            None,
            args.embedding_size,
            gen_args=gen_args,
        )
    else:
        model = TransformerModel(
            dictionary,
            args.embedding_size,
            args.nhead,
            args.hidden_size,
            args.nlayers,
            args.ngram,
            args.unk_threshold,
            gen_args=gen_args,
        )

    return model
