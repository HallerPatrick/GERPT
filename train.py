"""The main entry script for the pre-training tasks for LSTMs and Transformer"""

import os
from pathlib import Path


import torch
from datasets.load import load_from_disk
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from pytorch_lightning.profiler import PyTorchProfiler
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy

import wandb
from src import USE_CACHE
from src.args import parse_args, print_args, read_config, write_to_yaml
from src.dataset import GenericDataModule, load_tokenized_dataset, process_tokenized_dataset
from src.models import load_model
from src.models.transformer import NGMETokenizer
from src.utils import (
    ModelCheckpointCallback,
    TimePerEpochCallback,
    count_parameters,
)



from train_ds import train_ds

TEST = True

if __name__ == "__main__":

    # --- Init ---
    args = parse_args()

    # Show all configurations
    print_args(args)

    # Set seed
    torch.manual_seed(args.seed)

    tar_file = args.saved_data # + "-0.tar"

    # if os.path.isfile(tar_file):
    #     args.saved_data = tar_file

    if args.saved_dict and args.saved_data:
        print("saved_dict and saved_data not defined in config")

    print("Load preprocessed dataset from disk...")
    tokenized_dataset = load_tokenized_dataset(args.saved_data, args.ngram)

    dictionary = torch.load(args.saved_dict)

    # To avoid locks during distributed training
    wandb.require(experiment="service")

    # Init logger with all configs logged
    wandb_logger = WandbLogger(
        project="gerpt",
        offline=True,
        group=args.group,
        config={**vars(args), "dict_size": len(dictionary)},
    )

    # Init PL data module
    data_module = GenericDataModule(tokenized_dataset, args.batch_size, args.bptt, None, args.cpus)

    # --- PL Callbacks ---
    checkpoint_callback = ModelCheckpointCallback(
        monitor="train/loss",
        save_on_train_epoch_end=True,
        dirpath="checkpoints",
        filename=args.save,
    )

    # Peformance
    log_time_per_epoch_callback = TimePerEpochCallback()

    # Make it 🌟 pretty
    rick_prog_bar_callback = RichProgressBar()

    learning_rate_callback = LearningRateMonitor()

    # --- PL Plugins ---
    plugins = []

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
        logger=wandb_logger,
        max_epochs=args.epochs,
        # accelerator="auto",
        strategy=strategy,
        plugins=plugins,
        devices=args.gpus,
        gradient_clip_val=0.25,
        callbacks=[
            checkpoint_callback,
            # early_stop_callback,
            rick_prog_bar_callback,
            log_time_per_epoch_callback,
            learning_rate_callback,
        ],
        # Disable validation during training
        limit_val_batches=0.0,
        # profiler=PyTorchProfiler(),
        fast_dev_run=False,
    )

    model = load_model(dictionary, args)

    # Print Parameters
    if hasattr(model, "rnn"):
        print(count_parameters(model.rnn))
    else:
        print(count_parameters(model))

    # wandb_logger.log_metrics({"encoder_params": get_encoder_params(model)})

    # TRAIN!
    trainer.fit(model, data_module)
    
    if TEST:
        trainer.test(model, data_module)

    # Combine sharded model checkpoints into one for future loading
    if (
        strategy
        and "deepspeed_stage_" in strategy.strategy_name
        and hasattr(checkpoint_callback, "save_path")
    ):
        ckpt_path = checkpoint_callback.save_path
        print(f"Convert to single checkpoint: {ckpt_path}.single")
        convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, ckpt_path + ".single")

    # Custom save for flair embeddings
    if args.model == "lstm":
        model.save("checkpoints" / Path(args.save))

    # Transformer is wrapped in huggingface PreTrainedModel
    elif args.model == "transformer":

        trainer.lightning_module.model.save_pretrained(
            "checkpoints" / Path(args.save), ngram=args.ngram
        )

        # Save vocab file
        vocab_file = dictionary.save_vocabulary(
            "checkpoints" / Path(args.save), NGMETokenizer.vocab_file_name, args.ngram
        )
        
        # Save HF tokenizer
        NGMETokenizer(vocab_file).save_pretrained("checkpoints" / Path(args.save))

        # Save HF model

    try:
        # Save wandb run id in config for fine tuning run
        if hasattr(args, "wandb_flair_yaml") and args.wandb_flair_yaml:
            assert wandb.run is not None
            write_to_yaml(args.wandb_flair_yaml, "wandb_run_id", wandb.run.path)
    except:
        print("Could not write wandb RUN ID to flair config file")

    # Auto downstream training
    if hasattr(args, "downstream") and args.downstream:
        assert args.fine_tune_configs is not None
        assert wandb.run is not None
        fine_tune_args = read_config(args.fine_tune_configs)
        train_ds(fine_tune_args, wandb.run.path)
