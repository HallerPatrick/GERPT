import logging
import math
import os
import sys
import string
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, Mapping, Optional

import datasets
import evaluate
import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import (AdamW, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser,
                          Trainer, TrainingArguments,
                          default_data_collator, get_constant_schedule_with_warmup,
                          set_seed, AutoConfig)
from transformers.integrations import WandbCallback
from transformers.trainer_utils import get_last_checkpoint

import wandb
from src.data import local_dataset_mapper
from src.models import GPTNGMEConfig, GPTNGMETokenizer, CharFormerConfig
from src.models import NGMERwkvConfig, RwkvForCausalLM
from src.models.char_former.modelling_transformer import NextCharTransformerForCausalLM
from src.models.char_former.tokenization_transformer import CharacterTokenizer
from src.models.transformer.modelling_transformer import GPTNGMEForCausalLM

logger = logging.getLogger(__name__)

@dataclass
class ModelParamArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    hidden_size: int = field(
        default=512,
        metadata={"help": "Attention head hidden size"},
    )

    num_hidden_layers: int = field(
        default=2,
        metadata={
            "help": "Number of hidden layers."
        },
    )

    num_attention_heads: int = field(
        default=2,
        metadata={
            "help": "Number of attention heads."
        },
    )

    intermediate_size: int = field(
        default=512,
        metadata={
            "help": "FFN size"
        },
    )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    vocab_file: Optional[str] = field(
        default="./vocabs/3-gram-babylm.json",
        metadata={"help": "Path to vocab file"},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep line breaks when using TXT files or not."},
    )

    use_ngme: bool = field(
        default=True,
        metadata={"help": "Wether we are using N-Gram Mulithot Encoding"},
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`validation_file` should be a csv, a json or a txt file."


class TextGenerationCallback(WandbCallback):
    def __init__(self, tokenizer, model: GPTNGMEForCausalLM):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model

    def on_log(self, args: TrainingArguments, state, control, **kwargs):

        super().on_log(args, state, control, **kwargs)
        # Generate text and log it
        self.model.eval()

        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        # self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        input_prompt = "It might be possible to"

        input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids.to(
            self.model.device
        )

        # self.model.tokenizer = self.tokenizer

        output = self.model.sample(input_ids)
        
        # TODO: Pass max_length from config
        # text = self.model.sample(input_ids, tokenizer=self.tokenizer, max_length=2000, token_divider="·")
        print("Generated Output:")
        decoded = self.tokenizer.convert_ids_to_tokens(output.squeeze(0)[0])
        sequence = "".join(decoded)
        print(sequence)

        self.model.train()

        table = wandb.Table(
            columns=["global_step", "text"], data=[[state.global_step, str(sequence)]]
        )
        self._wandb.log({"text_generation": table})

class NumTokensCallback(WandbCallback):
    def __init__(self, num_processes: int, block_size: int):
        super().__init__()
        self.num_processes = num_processes
        self.block_size = block_size

    def on_log(self, args: TrainingArguments, state, control, **kwargs):
        if state.is_local_process_zero:
            g_step = state.global_step
            total_batch_size = args.per_device_train_batch_size * self.num_processes * args.gradient_accumulation_steps
            num_tokens = self.block_size * total_batch_size * g_step
            self._wandb.log({"num_tokens": num_tokens})


def ngme_data_collator(features) -> Dict[str, Any]:
    # features: list

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = (
            first["label"].item()
            if isinstance(first["label"], torch.Tensor)
            else first["label"]
        )
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor(
                [f["label_ids"] for f in features], dtype=dtype
            )

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                # features[0][k]: [ngram, seq_len]
                batch[k] = torch.stack([torch.tensor(f[k]) for f in features], dim=0)
                # batch[k]: [batch_size, ngram, seq_len]

    return batch


def main():

    parser = HfArgumentParser((DataTrainingArguments, TrainingArguments, ModelParamArguments))

    data_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # download the dataset.
    if data_args.dataset_name.startswith("babylm"):
        local_dataset = local_dataset_mapper[data_args.dataset_name]
        raw_datasets = load_dataset("text", data_files=local_dataset, keep_linebreaks=True)

    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                **dataset_args,
            )

    if data_args.use_ngme:
        tokenizer = GPTNGMETokenizer(vocab_file=data_args.vocab_file)
        # config = GPTNGMEConfig(
        #     vocab_size=tokenizer.vocab_size,
        #     hidden_size=model_args.hidden_size,
        #     num_hidden_layers=model_args.num_hidden_layers,
        #     num_attention_heads=model_args.num_attention_heads,
        #     intermediate_size=model_args.intermediate_size,
        #     use_ngme=data_args.use_ngme,
        #     eos_token_id=tokenizer.eos_token_id,
        #     unk_token_id=tokenizer.unk_token_id,
        #     pad_token_id=tokenizer.pad_token_id
        # )

        config = NGMERwkvConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=model_args.hidden_size,
            num_hidden_layers=model_args.num_hidden_layers,
            intermediate_size=model_args.intermediate_size,
            unk_token_id=tokenizer.unk_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    else:
        chars = string.ascii_letters + " "# This character vocab!
        model_max_length = 2048
        tokenizer = CharacterTokenizer(chars, model_max_length)
        # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        # config = AutoConfig.from_pretrained("EleutherAI/pythia-70m")

        config = CharFormerConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=model_args.hidden_size,
            num_hidden_layers=model_args.num_hidden_layers,
            num_attention_heads=model_args.num_attention_heads,
            intermediate_size=model_args.intermediate_size,
            use_ngme=data_args.use_ngme,
            unk_token_id=tokenizer.unk_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    model = AutoModelForCausalLM.from_config(config)

    print(model)

    if hasattr(model, "set_tokenizer"):
        model.set_tokenizer(tokenizer)

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(
        f"Training new model from scratch - Total size={n_params/2**20:.2f}M params"
    )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )

    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])
        # output input_ids should be same amount as examples to tokenizer

        if data_args.use_ngme:
            assert len(examples[text_column_name]) == len(output["input_ids"])
            # Attention mask should be as long as first input ids
            if len(output["input_ids"][0]) > 0:
                assert len(output["input_ids"][0][0]) == len(
                    output["attention_mask"][0]
                )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
                drop_last_batch=True,
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            logger.warning(f"Tokenizer block size is {block_size}")
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.

        concatenated_examples = {}
        for k in examples.keys():
            if data_args.use_ngme and k == "input_ids":
                concatenated_examples[k] = np.concatenate(
                    [np.array(example) for example in examples[k] if len(example) > 0],
                    axis=1,
                )
            else:
                concatenated_examples[k] = list(chain(*examples[k]))

        total_length = len(concatenated_examples["attention_mask"])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        result = {}

        # Split by chunks of max_len.
        for k, t in concatenated_examples.items():
            if k == "input_ids" and data_args.use_ngme:
                result[k] = [
                    t[:, i : i + block_size] for i in range(0, total_length, block_size)
                ]
            else:
                result[k] = [
                    t[i : i + block_size] for i in range(0, total_length, block_size)
                ]

        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels

            # Accuray over unigrams
            labels = labels[:, 0, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)

            return metric.compute(predictions=preds, references=labels)
    
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        optimizers=(optimizer, get_constant_schedule_with_warmup(optimizer, 300)),
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=ngme_data_collator
        if data_args.use_ngme
        else default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval
        else None,
    )

    trainer.add_callback(TextGenerationCallback(tokenizer, model))
    # trainer.add_callback(NumTokensCallback(trainer.args.world_size, block_size))

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
