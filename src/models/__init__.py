from argparse import Namespace
from typing import Dict

from src.dataset import Dictionary
from src.models.rnn import RNNModel
from src.models.transformer import TransformerModel
from src.utils import calculate_lstm_hidden_size


def load_model(dictionary: Dictionary, args: Namespace, gen_args: Dict):

    if hasattr(args, "expected_size"):
        if args.expected_size > 0:
            new_hidden_size = int(
                calculate_lstm_hidden_size(
                    len(dictionary),
                    args.embedding_size,
                    4,
                    args.nlayers,
                    args.expected_size,
                    args.hidden_size,
                )
            )
            print(f"New Hidden Size: {new_hidden_size}")
            args.hidden_size = new_hidden_size

    if args.model == "lstm":
        model = RNNModel(
            dictionary,
            args.nlayers,
            args.ngram,
            args.hidden_size,
            args.unk_threshold,
            None,
            args.embedding_size,
            gen_args=gen_args,
            unigram_ppl=args.unigram_ppl,
            weighted_loss=args.weighted_loss,
            weighted_labels=args.weighted_labels,
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
            unigram_ppl=args.unigram_ppl,
            weighted_loss=args.weighted_loss,
            weighted_labels=args.weighted_labels,
            n_pos_embeddings=args.n_pos_embeddings,
        )

    return model
