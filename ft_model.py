from typing import Any, Dict, Mapping
from datasets import load_dataset

import numpy as np
import evaluate
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from src import models

MODEL_PATH = "/tmp/test-clm"
max_sequence_length = 128

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

    training_args = TrainingArguments(output_dir="test_trainer", num_train_epochs=10, evaluation_strategy="epoch")

    dataset = load_dataset("yelp_review_full")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_sequence_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = (
        tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    )
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=5)

    metric = evaluate.load("accuracy")

    # def compute_metrics(eval_pred):
    #     # logits, labels = eval_pred
    #     # predictions = np.argmax(logits, axis=-1)
    #
    #     preds, labels = eval_pred
    #     # preds have the same shape as the labels, after the argmax(-1) has been calculated
    #     # by preprocess_logits_for_metrics but we need to shift the labels
    #
    #     # Accuray over unigrams
    #     labels = labels[:, 0, 1:].reshape(-1)
    #     preds = preds[:, :-1].reshape(-1)
    #     return metric.compute(predictions=preds, references=labels)

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=ngme_data_collator,
    )

    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()
