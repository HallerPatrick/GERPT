from argparse import Namespace
from typing import Dict

from src.dataset import Dictionary
from src.models.rnn import RNNModel
from src.models.transformer.configuration_transformer import TransformerConfig
from src.models.transformer.transformer import TransformerLightningModule

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
        # Adjust args
        args.ntoken = len(dictionary)
    #     print(args.ntoken)
    #     print(dictionary.word2idx)
    #     print(dictionary.idx2word)
    # 
    #     exit()
        if args.weighted_loss:
            args.weight_tensor = dictionary.create_weight_tensor()

        model = TransformerLightningModule(
            TransformerConfig.from_args(args)
        )

    return model
