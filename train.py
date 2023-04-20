"""The main entry script for the pre-training tasks for LSTMs and Transformer"""

from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

import wandb
from src.args import parse_args, print_args, read_config, write_to_yaml
from src.models import load_model
from src.dictionary import Dictionary

from src.train_strategy import TrainStrategy
from src.utils import (
    FlairDownstreamCallback,
    TextGenerationCallback,
    TimePerEpochCallback,
    EarlyStoppingOnLRCallback,
)
from train_ds import train_ds

TEST = False

torch.set_float32_matmul_precision("medium")

def pl_callbacks():
    """Return a list of callbacks for the trainer"""
    checkpoint_callback = ModelCheckpoint(
        monitor="epoch",
        every_n_epochs=10,
        mode="max",
        dirpath="checkpoints",
        filename=args.save,
        verbose=True,
    )

    # Peformance
    log_time_per_epoch_callback = TimePerEpochCallback()

    # Make it 🌟 pretty
    rick_prog_bar_callback = RichProgressBar()

    learning_rate_callback = LearningRateMonitor()

    early_stopping_callback = EarlyStoppingOnLRCallback(lr_threshold=0.01)

    text_gen_callback = TextGenerationCallback(interval=1, enabled=False)

    flair_callback = FlairDownstreamCallback(interval=10, enabled=False)

    return [
        checkpoint_callback,
        log_time_per_epoch_callback,
        rick_prog_bar_callback,
        learning_rate_callback,
        early_stopping_callback,
        text_gen_callback,
        flair_callback
    ]


if __name__ == "__main__":
    # --- Init ---
    args = parse_args()

    # Show all configurations
    print_args(args)

    # Set seed
    torch.manual_seed(args.seed)

    if not args.saved_dict or not args.saved_data:
        print("saved_dict and saved_data not defined in config")
        exit()

    # To avoid locks during distributed training
    wandb.require(experiment="service")

    # Init logger with all configs logged
    wandb_logger = WandbLogger(
        project="gerpt",
        offline=not args.online,
        group=args.group,
        config={**vars(args)},
    )

    if torch.cuda.is_available():
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = None

    # --- Load Dictionary & Model ---
    if args.saved_dict.endswith(".json"):
        dictionary = Dictionary.load_from_file(args.saved_dict)
        dictionary.ngme = "explicit"
        dictionary = dictionary.unking(100_000, 4, 2000, True)
    else:
        dictionary = torch.load(args.saved_dict)

    model = load_model(dictionary, args, print_params=True)

    # --- Training ---
    trainer = Trainer(
        # resume_from_checkpoint=args.continue_from
        # if args.continue_from and Path(args.continue_from).exists()
        # else None,
        logger=wandb_logger,
        max_epochs=args.epochs,
        strategy=strategy,
        accelerator="auto",
        devices=args.gpus,
        precision=16,
        gradient_clip_val=0.25,
        callbacks=pl_callbacks(),
        # Disable validation during training
        limit_val_batches=0.0,
        fast_dev_run=False,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=10,
    )

    # Train model based on data module and strategy
    TrainStrategy.train_from_strategy(trainer, model, args)

    # --- Save ---
    last_ckpt_path = "checkpoints/" + args.save + ".last.ckpt"

    trainer.save_checkpoint(last_ckpt_path)

    # Custom save for flair embeddings
    if args.model == "lstm":
        model.save("checkpoints" / Path(args.save))

    # Transformer is wrapped in huggingface PreTrainedModel
    elif args.model == "transformer":
        pass
        # trainer.lightning_module.model.save_pretrained(
        #     "checkpoints" / Path(args.save), ngram=args.ngram
        # )
        #
        # if not args.reuse_dict:
        #     # Save vocab file
        #     vocab_file = dictionary.save_vocabulary(
        #         "checkpoints"
        #         / Path(args.save)
        #         / (NGMETokenizer.vocab_file_name + ".dict"),
        #         args.ngram,
        #     )
        # else:
        #     vocab_file = args.reuse_dict
        #
        # # Save HF tokenizer
        # NGMETokenizer(vocab_file).save_pretrained("checkpoints" / Path(args.save))

    try:
        # Save wandb run id in config for fine tuning run
        if hasattr(args, "wandb_flair_yaml") and args.wandb_flair_yaml:
            assert wandb.run is not None
            write_to_yaml(args.wandb_flair_yaml, "wandb_run_id", wandb.run.path)
    except:
        print("Could not write wandb RUN ID to flair config file")

    print("Training done")
    print("=" * 80)
    print("Saving:")
    print(f"Flair Model:     {'checkpoints' / Path('flair_' + args.save)}")
    print(
        f"Last Checkpoint: {last_ckpt_path}{'(deepspeed folder)' if strategy else ''}"
    )
    if strategy:
        print(
            f"Solidified deepspeed Checkpoint:                     {last_ckpt_path + '.single'}"
        )
    print("=" * 80)

    # Auto downstream training
    if hasattr(args, "downstream") and args.downstream:
        assert args.fine_tune_configs is not None
        assert wandb.run is not None
        fine_tune_args = read_config(args.fine_tune_configs)
        train_ds(fine_tune_args, wandb.run.path)
