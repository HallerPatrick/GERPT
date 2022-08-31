import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import yaml
from rich.console import Console
from rich.table import Table


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

    return yaml_args


def print_args(args):
    """Pretty print all arguments/hyperparameter

    Args:
        args: Namespace object
    """
    table = Table(title="Configurations")
    table.add_column(" ", no_wrap=True)
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="cyan", no_wrap=True)

    for i, (parameter, value) in enumerate(args.__dict__.items()):
        table.add_row(str(i), parameter, str(value))

    console = Console()
    console.print(table)


def read_config(path):
    """Return namespace object like argparser of yaml file"""

    with open(path, "r") as f:
        conf = yaml.safe_load(f)

    return json.loads(json.dumps(conf), object_hook=lambda d: SimpleNamespace(**d))


def write_to_yaml(path_name: str, param: str, value: str):
    """
    Read in yaml file and write in new key-value

    Args:
        path_name: path to yaml file
        param: key
        value: value
    """
    path = Path(path_name)

    conf = {}

    if path.exists():
        with open(str(path), "r") as f:
            conf = yaml.safe_load(f)
    conf[param] = value

    with open(str(path), "w") as f:
        yaml.dump(conf, f)


def argparser_train() -> argparse.Namespace:
    """Argparser for the pre-training task

    Returns:
        Namespace
    """
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Configration file (YAML) for all arguments, if empty, use command lines arguments",
    )
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
    parser.add_argument(
        "--weighted-loss",
        action="store_true",
        help="Use a weighted loss for n-gram",
    )
    parser.add_argument(
        "--weighted-labels",
        action="store_true",
        help="Use a weighted target labels for n-gram",
    )
    # parser.add_argument(
    #     "--unigram-ppl",
    #     action="store_true",
    #     help="Calculate perplexity only over unigrams",
    # )
    parser.add_argument("--embedding-size", type=int, help="size of word embeddings")
    parser.add_argument(
        "--expected-size", type=int, help="Size of model adjusted by tuning hidden size"
    )
    parser.add_argument(
        "--hidden-size", type=int, help="number of hidden units per layer"
    )
    parser.add_argument("--nlayers", type=int, help="number of layers")
    parser.add_argument("--lr", type=float, help="initial learning rate")
    parser.add_argument("--clip", type=float, help="gradient clipping")
    parser.add_argument("--epochs", type=int, help="upper epoch limit")
    parser.add_argument("--batch-size", type=int, metavar="N", help="batch size")
    parser.add_argument("--bptt", type=int, help="sequence length")
    parser.add_argument(
        "--dropout",
        type=float,
        help="dropout applied to layers (0 = no dropout)",
    )
    parser.add_argument("--gpus", type=int, help="number of gpus used")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--save", type=str, help="path to save the final model")
    parser.add_argument(
        "--nhead",
        type=int,
        help="the number of heads in the encoder/decoder of the transformer model",
    )

    return parser.parse_args()


def argparse_flair_train() -> argparse.Namespace:
    """Argparser for the downstream training task(s)

    Returns:
        Namespace
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Configration file (YAML) for all arguments",
    )

    parser.add_argument(
        "--saved_model",
        type=str,
    )

    return parser.parse_args()


def argparse_generate():
    """Argparser for standalone generator"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
    )
    parser.add_argument(
        "--temperature",
        type=float,
    )

    parser.add_argument(
        "--type",
        type=str,
        help="Either transformer or rnn"
    )

    parser.add_argument("--mode", type=str)

    parser.add_argument("--num", type=int, default=1)
    return parser.parse_args()
