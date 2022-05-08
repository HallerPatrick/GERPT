from argparse import Namespace
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from torch.utils.data import DataLoader
from datasets import load_dataset

from src.models.rnn import RNNModel
from src.dataset import load_dictionary_from_hf


args = {
        "ngram": 2,
        "max_dict_size": 0,
        "unk_threshold": 5
}

args = Namespace(**args)

# wandb_logger = WandbLogger(project="gerpt")

dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

load_dictionary_from_hf(dataset, args.ngram, args.max_dict_size, args.unk_threshold)
# print(dataset)

# dataset = dataset.map(lambda x: print(x))

# print(dataset["test"]["text"])

# trainer = Trainer(logger=wandb_logger)

# dataloader = DataLoader(dataset)

# model = RNNModel()
# trainer.fit(model, dataloader)

