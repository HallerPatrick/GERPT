import logging
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin

import wandb
from src.args import parse_args, print_args, write_to_yaml
from src.dataset import GenericDataModule, load_tokenized_dataset
from src.models import load_model

if __name__ == "__main__":

    # --- Init ---
    args = parse_args()

    gen_args = {"generate": True, "chars": 1000, "temperature": 0.7}

    wandb_logger = WandbLogger(project="gerpt", offline=True)
    wandb_logger.experiment.config.update(vars(args))

    print_args(args)

    # --- Dataloading & Tokenization ---
    tokenized_dataset, dictionary = load_tokenized_dataset(
        args.bptt,
        args.ngram,
        args.max_dict_size,
        args.unk_threshold,
        args.fallback,
        args.cpus,
        *args.data.split("/")
    )

    data_module = GenericDataModule(tokenized_dataset, args.batch_size, args.cpus)

    # --- PL Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        monitor="train/loss",
        save_on_train_epoch_end=True,
        dirpath="checkpoints",
        filename=args.save + "-{epoch:02d}",
    )

    early_stop_callback = EarlyStopping(
        monitor="train/loss", patience=3, verbose=True, mode="min"
    )

    # Make it 🌟 pretty
    rick_prog_bar_callback = RichProgressBar()
    rich_model_summary_callback = RichModelSummary()

    # --- PL Plugins ---
    plugins = []
    if torch.cuda.is_available():
        # plugins.append(DeepSpeedPlugin(logging_level=logging.DEBUG))
        # plugins.append(MixedPrecisionPlugin())
        pass


    # --- Training ---
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=args.epochs,
        accelerator="auto",
        # strategy="deepspeed",
        plugins=plugins,
        precision=16,
        devices=args.gpus,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            rick_prog_bar_callback,
            rich_model_summary_callback,
        ],
        # Disable validation during training
        limit_val_batches=0.0,
        profiler="simple",
    )

    model = load_model(dictionary, args, gen_args)

    trainer.fit(model, data_module)

    # Custom save for flair
    model.save("checkpoints" / Path(args.save))

    # Save wandb run id in config for fine tuning run
    if hasattr(args, "wandb_flair_yaml") and args.wandb_flair_yaml:
        write_to_yaml(args.wandb_flair_yaml, "wandb_run_id", wandb.run.path)
