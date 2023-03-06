from argparse import Namespace
from pathlib import Path

from src.models.rnn import RNNModel
from src.models.transformer.configuration_transformer import TransformerConfig
from src.models.transformer.transformer import TransformerLightningModule
from src.utils import count_parameters
from src.models import llama


def load_model(dictionary, args: Namespace, print_params: bool = True):
    if "lstm" in args.model:
        if args.continue_from and Path(args.continue_from).exists():
            print("LOADING FROM CHECKPOINT")
            model = RNNModel.load_from_checkpoint(args.continue_from, lr=args.lr)
        else:
            model = RNNModel(
                dictionary,
                args.nlayers,
                args.ngram,
                args.hidden_size,
                None,
                args.embedding_size,
                args.lr,
                unigram_ppl=args.unigram_ppl,
                weighted_loss=args.weighted_loss,
                weighted_labels=args.weighted_labels,
                strategy=args.weight_strat,
                generate=args.generate,
                temperature=args.temperature,
                chars_to_gen=args.chars,
                is_forward_lm=args.is_forward,
                cell_type=args.model,
                packed=args.packed,
            )
    else:
        # Adjust args
        args.ntoken = len(dictionary)
        args.weight_tensor = dictionary.create_weight_tensor(
            args.unigram_ppl, args.weighted_loss
        ).tolist()

        args.ngram_indexes = dictionary.ngram_indexes
        # args.pad_token_id = dictionary.word2idx["<pad>"]

        model = TransformerLightningModule(
            TransformerConfig.from_args(args), dictionary=dictionary
        )

    if print_params:
        # Print Parameters
        if hasattr(model, "rnn"):
            print(count_parameters(model.rnn))
        else:
            print(count_parameters(model))

    return model
