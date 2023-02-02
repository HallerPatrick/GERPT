"""
Some sort of eval harness for the BabyLM Challenge (https://babylm.github.io/index.html).
Model is pretrained like usual (see README).
"""

from pytorch_lightning import Trainer
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy

import torch

from src.args import argparse_babylm
from src.dataset import GenericDataModule
from src.models.rnn import RNNModel


def main():

    args = argparse_babylm()

    dictionary = torch.load(args.dict)
    model = RNNModel.load_from_checkpoint(args.model)
    dataset = torch.load(args.dataset)

    data_module = GenericDataModule(dataset, args.batch_size, args.bptt, None, args.cpus)

    if torch.cuda.is_available():
        strategy = DeepSpeedStrategy(
            accelerator="auto",
            logging_batch_size_per_gpu=args.batch_size
        )
        # strategy = "deepspeed_stage_2"
    else:
        strategy = None

    # --- Training ---
    trainer = Trainer(
        max_epochs=args.epochs,
        devices=args.gpus,
        strategy=strategy
    )

    trainer.test(model, data_module)



if __name__ == '__main__':
    main()
