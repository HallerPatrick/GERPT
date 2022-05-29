from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import RichProgressBar

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from src.args import parse_args, print_args, write_to_yaml
from src.models import load_model

from src.dataset import GenericDataModule, load_tokenized_dataset


if __name__ == "__main__":

    # --- Init ---
    args = parse_args()

    gen_args = {"generate": True, "chars": 1000, "temperature": 0.0}

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

    rick_prog_bar_callback = RichProgressBar()

    # --- Training ---
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=args.epochs,
        accelerator="auto",
        strategy="deepspeed",
        devices=args.gpus,
        callbacks=[checkpoint_callback, early_stop_callback, rick_prog_bar_callback],
        # Disable validation during training
        limit_val_batches=0.0,
    )

    model = load_model(dictionary, args, gen_args)

    trainer.fit(model, data_module)

    # Custom save for flair
    model.save("checkpoints" / Path(args.save))

    # Save wandb run id in config for fine tuning run
    if hasattr(args, "wandb_flair_yaml") and args.wandb_flair_yaml:
        write_to_yaml(args.wandb_flair_yaml, "wandb_run_id", wandb.run.path)
