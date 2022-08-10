from pathlib import Path
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)

import wandb
from train_ds import train_ds
from src.args import parse_args, print_args, read_config, write_to_yaml
from src.dataset import GenericDataModule, load_tokenized_dataset
from src.models import load_model
from src.utils import ModelCheckpointCallback, TimePerEpochCallback, get_encoder_params

from src.models.transformer import NGMETokenizer

if __name__ == "__main__":

    # --- Init ---
    args = parse_args()

    gen_args = {"generate": True, "chars": 1000, "temperature": 0.7}

    print_args(args)

    # --- Dataloading & Tokenization ---
    tokenized_dataset, dictionary = load_tokenized_dataset(
        args.bptt,
        args.ngram,
        args.max_dict_size,
        args.unk_threshold,
        args.fallback,
        args.cpus,
        *args.data.split("/"),
    )

    wandb.require(experiment="service")
    configs = {**vars(args), "dict_size": len(dictionary)}
    wandb_logger = WandbLogger(project="gerpt", offline=True, config=configs)

    dset_metrics = {}

    dset_metrics[f"total_tokens"] = sum(dictionary.total_n_tokens.values()) + sum(
        dictionary.unk_n_tokens.values()
    )
    for n in range(1, args.ngram + 1):
        dset_metrics[f"total_{n}_gram_tokens"] = dictionary.total_n_tokens[n]
        dset_metrics[f"total_{n}_gram_unk_tokens"] = dictionary.unk_n_tokens[n]

    wandb_logger.log_metrics(dset_metrics)

    data_module = GenericDataModule(tokenized_dataset, args.batch_size, args.cpus)

    # --- PL Callbacks ---
    checkpoint_callback = ModelCheckpointCallback(
        monitor="train/loss",
        save_on_train_epoch_end=True,
        dirpath="checkpoints",
        filename=args.save,
    )

    early_stop_callback = EarlyStopping(
        monitor="train/loss", patience=3, verbose=True, mode="min"
    )

    # Make it 🌟 pretty
    rick_prog_bar_callback = RichProgressBar()

    log_time_per_epoch_callback = TimePerEpochCallback()

    # --- PL Plugins ---
    plugins = []
    if torch.cuda.is_available():
        pass

    strategy = "deepspeed_stage_2"
    strategy = None

    # --- Training ---
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=args.epochs,
        accelerator="auto",
        strategy=strategy,
        plugins=plugins,
        devices=args.gpus,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            rick_prog_bar_callback,
            log_time_per_epoch_callback,
        ],
        # Disable validation during training
        limit_val_batches=0.0,
        # profiler="simple",
        # fast_dev_run=True
    )

    model = load_model(dictionary, args, gen_args)

    if hasattr(model, "rnn"):
        for name, parameter in model.rnn.named_parameters():
            if parameter.requires_grad:
                print(name, parameter.numel())

    wandb_logger.log_metrics({"encoder_params": get_encoder_params(model)})

    # TRAIN!
    trainer.fit(model, data_module)

    # Combine sharded model checkpoints into one for future loading
    if strategy and "deepspeed_stage_" in strategy:
        ckpt_path = checkpoint_callback.save_path
        print(f"Convert to single checkpoint: {ckpt_path}.single")
        convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, ckpt_path + ".single")

    # Custom save for flair embeddings
    if args.model == "lstm":
        model.save("checkpoints" / Path(args.save))

    # Transformer is wrapped in huggingface PreTrainedModel
    elif args.model == "transformer":
        # Save vocab file
        vocab_file = dictionary.save_vocabulary(
            "checkpoints", NGMETokenizer.vocab_file_name
        )

        # Save HF model
        trainer.lightning_module.model.save_pretrained("checkpoints" / Path(args.save))
        # NGMETokenizer(trainer.lightning_module.model.config)

        # Load HF tokenizer and save it for downstream with flair
        NGMETokenizer(vocab_file).save_pretrained("checkpoints" / Path(args.save))

    try:
        # Save wandb run id in config for fine tuning run
        if hasattr(args, "wandb_flair_yaml") and args.wandb_flair_yaml:
            write_to_yaml(args.wandb_flair_yaml, "wandb_run_id", wandb.run.path)
    except:
        pass

    # Auto downstream training
    if hasattr(args, "fine_tune") and args.fine_tune:
        assert args.fine_tune_configs is not None
        fine_tune_args = read_config(args.fine_tune_configs)
        train_ds(fine_tune_args, wandb.run.path)
