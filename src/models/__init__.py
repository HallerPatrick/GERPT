from argparse import Namespace
from typing import Dict

from src.models.rnn import RNNModel
from src.models.transformer.configuration_transformer import TransformerConfig
from src.models.transformer.transformer import TransformerLightningModule
from src.utils import calculate_lstm_hidden_size


def load_model(dictionary, args: Namespace):

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
            unigram_ppl=args.unigram_ppl,
            weighted_loss=args.weighted_loss,
            weighted_labels=args.weighted_labels,
            strategy=args.weight_strat,
            generate=args.generate,
            temperature=args.temperature,
            chars_to_gen=args.chars,
            is_forward_lm=args.is_forward
        )
    else:
        # Adjust args
        args.ntoken = len(dictionary)
        args.weight_tensor = dictionary.create_weight_tensor(args.unigram_ppl, args.weighted_loss).tolist()

        args.ngram_indexes = dictionary.ngram_indexes
        # args.pad_token_id = dictionary.word2idx["<pad>"]

        model = TransformerLightningModule(
            TransformerConfig.from_args(args), dictionary=dictionary
        )

    return model
