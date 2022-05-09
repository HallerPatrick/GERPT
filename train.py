from argparse import Namespace
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from torch.utils.data import DataLoader
from datasets import load_dataset

from src.models.rnn import RNNModel
from src.dataset import load_tokenized_dataset


args = {
    "ngram": 2,
    "max_dict_size": 0,
    "unk_threshold": 5,
    # "data": "wikitext/wikitext-103-raw-v1",
    "data": "wikitext/wikitext-2-raw-v1",
    # "data": "bookcorpus",
    "fallback": True,
    "nlayers": 2,
    "hidden_size": 124,
    "embedding_size": 124,
}

args = Namespace(**args)

# wandb_logger = WandbLogger(project="gerpt")

tokenized_dataset, dictionary = load_tokenized_dataset(
    args.ngram,
    args.max_dict_size,
    args.unk_threshold,
    args.fallback,
    *args.data.split("/")
)

dataloader = DataLoader(tokenized_dataset["train"])

model = RNNModel(
    dictionary,
    args.nlayers,
    args.ngram,
    args.hidden_size,
    args.unk_threshold,
    None,
    args.embedding_size,
)
trainer = Trainer()
trainer.fit(model, dataloader)
