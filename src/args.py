import argparse
from argparse import Namespace

from prettytable.prettytable import PrettyTable

import yaml

def parse_args():
    """Params can be set through a config yaml and/or passed as arguments.
    A argument will overwrite a config parameter.
    """

    args = argparser_train()

    if not args.config:
        return args

    yaml_args = read_config(args.config)

    for key, value in args.__dict__.items():
        if value:
            yaml_args.__dict__.update({key: value})

    
    table = PrettyTable([ "Parameter", "Value" ])
    
    for parameter, value in yaml_args.__dict__.items():
        table.add_row([parameter, value])

    print("Configurations:")
    print(table)

    return yaml_args

def read_config(path):
    """Return namespace object like argparser of yaml file"""

    with open(path, "r") as f:
        conf = yaml.safe_load(f)
    
    return Namespace(**conf)

def argparser_generate():

    parser = argparse.ArgumentParser(description="PyTorch Wikitext-2 Language Model")

    # Model parameters.
    parser.add_argument(
        "--data",
        type=str,
        help="location of the data corpus",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="./model.pt", help="model checkpoint to use"
    )
    parser.add_argument(
        "--outf",
        type=str,
        default="generated.txt",
        help="output file for generated text",
    )
    parser.add_argument(
        "--chars", type=int, default="1000", help="number of words to generate"
    )
    parser.add_argument(
        "--unk-threshold", type=int, default=3, help="UNK threshold for bigrams"
    )
    parser.add_argument("--ngram", type=int, default=2, help="N-Grams used")
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument("--cuda", action="store_true", help="use CUDA")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="temperature - higher will increase diversity",
    )
    parser.add_argument(
        "--log-interval", type=int, default=100, help="reporting interval"
    )
    return parser.parse_args()


def argparser_train():

    parser = argparse.ArgumentParser(
        description="PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model"
    )
    parser.add_argument("--config", type=str, default="", help="Configration file (YAML) for all arguments, if empty, use command lines arguments")
    parser.add_argument(
        "--data",
        type=str,
        help="location of the data corpus",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)",
    )
    parser.add_argument(
        "--unk-threshold", type=int, default=3, help="UNK threshold for bigrams"
    )

    parser.add_argument(
        "--max-dict-size",
        type=int,
        default=0,
        help="Set max dictionary size, other tokens become UNK",
    )

    parser.add_argument("--ngram", type=int, help="N-Grams used")
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Fallback on n-1-gram if UNK for n-gram",
    )
    # parser.add_argument(
    #     "--unigram-ppl",
    #     action="store_true",
    #     help="Calculate perplexity only over unigrams",
    # )
    parser.add_argument(
        "--embedding-size", type=int, help="size of word embeddings"
    )
    parser.add_argument(
        "--hidden-size", type=int, help="number of hidden units per layer"
    )
    parser.add_argument("--nlayers", type=int, help="number of layers")
    parser.add_argument("--lr", type=float, help="initial learning rate")
    parser.add_argument("--clip", type=float, help="gradient clipping")
    parser.add_argument("--epochs", type=int, help="upper epoch limit")
    parser.add_argument(
        "--batch-size", type=int, metavar="N", help="batch size"
    )
    parser.add_argument("--bptt", type=int, help="sequence length")
    parser.add_argument(
        "--dropout",
        type=float,
        help="dropout applied to layers (0 = no dropout)",
    )
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument(
        "--save", type=str, help="path to save the final model"
    )
    parser.add_argument(
        "--nhead",
        type=int,
        help="the number of heads in the encoder/decoder of the transformer model",
    )

    return parser.parse_args()
