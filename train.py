from argparse import Namespace

import torch

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from torch.utils.data import DataLoader
from datasets import load_dataset

from src.models.rnn import RNNModel
from src.dataset import load_tokenized_dataset


args = {
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
    "epochs": 20,
    "cpus": 1,
    "gpus": 1,
}

gen_args = {
    "generate": True,
    "chars": 1000,
    "temperature": 0.0
}

args = Namespace(**args)

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


def batch_collate(batch):
    # [ngram, seq_len, batch_size]
    source = torch.cat([torch.tensor(t["source"]).unsqueeze(-1) for t in batch], dim=-1)
    target = torch.cat([torch.tensor(t["target"]).unsqueeze(-1) for t in batch], dim=-1)
    return dict(source=source, target=target)


dataloader = DataLoader(
    tokenized_dataset["train"],
    batch_size=args.batch_size,
    collate_fn=batch_collate,
    drop_last=True,
    # num_workers=args.cpus
)

model = RNNModel(
    dictionary,
    args.nlayers,
    args.ngram,
    args.hidden_size,
    args.unk_threshold,
    None,
    args.embedding_size,
    gen_args=gen_args
)
# trainer = Trainer(logger=wandb_logger, log_every_n_steps=10, max_epochs=args.epochs, accelerator="auto", devices=args.gpus)
trainer = Trainer(logger=wandb_logger, max_epochs=args.epochs, accelerator="auto", devices=args.gpus)
trainer.fit(model, dataloader)
