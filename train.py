from argparse import Namespace

import torch

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from src.models.rnn import RNNModel
from src.dataset import GenericDataModule, load_tokenized_dataset
from src.models.transformer import TransformerModel


args = {
    # "model": "transformer",
    "model": "rnn",
    "ngram": 1,
    "max_dict_size": 0,
    "unk_threshold": 0,
    # "data": "wikitext/wikitext-103-raw-v1",
    # "data": "wikitext/wikitext-2-raw-v1",
    "data": "text/cash",
    "fallback": False,
    "nlayers": 2,
    "hidden_size": 200,
    "embedding_size": 124,
    "batch_size": 1,
    "bptt": 200,
    "nhead": 2,
    "epochs": 20,
    "cpus": 1,
    "gpus": 1,
}

args = Namespace(**args)

gen_args = {"generate": True, "chars": 1000, "temperature": 0.0}

if __name__ == "__main__":

    wandb_logger = WandbLogger(project="gerpt", offline=True)
    wandb_logger.experiment.config.update(vars(args))

    tokenized_dataset, dictionary = load_tokenized_dataset(
        args.bptt,
        args.ngram,
        args.max_dict_size,
        args.unk_threshold,
        args.fallback,
        *args.data.split("/")
    )

    data_module = GenericDataModule(tokenized_dataset, args.batch_size, args.cpus)

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
            gen_args=gen_args
        )

    trainer = Trainer(
        logger=wandb_logger, max_epochs=args.epochs, accelerator="auto", devices=args.gpus
    )
    trainer.fit(model, data_module)
