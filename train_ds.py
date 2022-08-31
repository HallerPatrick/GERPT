from argparse import Namespace
from typing import Optional

from flair import set_seed
from flair.embeddings.document import DocumentRNNEmbeddings
from torch import manual_seed

import wandb
from src.args import argparse_flair_train, read_config
from src.models.flair_models import (NGMETransformerWordEmbeddings,
                                     load_corpus, patch_flair)


def train_ds(args: Optional[Namespace] = None, wandb_run_id: Optional[str] = None):
    """Downstream training procedure.

    Args:
        args(Namespace): Either directly read from config and command line or passed from pre-training
        wandb_run_id(Optional[str]): If passed from pre-training take existing wandb run id for logging
    """

    if not args:
        args = read_config(argparse_flair_train().config)

    # Seed everything
    set_seed(int(args.seed))
    manual_seed(int(args.seed))

    # Has to be called first, before importing flair modules
    if args.model_name == "rnn":
        patch_flair()

    from flair.embeddings import FlairEmbeddings, StackedEmbeddings
    from flair.models import SequenceTagger, TextClassifier
    from flair.trainers import ModelTrainer

    # Try to log to existing wandb run
    if wandb_run_id:
        args.wandb_run_id = wandb_run_id
        api = wandb.Api()
        try:
            run = api.run(args.wandb_run_id)
        except:
            print("Could not log to wandb run")
            run = None
    else:
        run = None

    results = {}

    for task_settings in args.downstream_tasks:

        settings = task_settings.task

        if not settings.use:
            print(f"Skipping fine-tune task {settings.task_name}")
            continue

        if run:
            run.config.update({settings.task_name: vars(settings)})
            run.update()

        corpus = load_corpus(settings.dataset, settings.base_path)

        label_dict = corpus.make_label_dictionary(label_type=settings.task_name)

        if settings.task_name in ["ner", "upos"]:

            if args.model_name == "rnn":
                embeddings = StackedEmbeddings(
                    embeddings=[FlairEmbeddings(args.saved_model)]
                )
            else:
                embeddings = NGMETransformerWordEmbeddings(
                    args.saved_model,
                    vocab_file=args.saved_model + "/vocab.txt",
                    layers="all",
                    subtoken_pooling="first",
                    fine_tune=False,
                    use_context=False,
                )

            task_model = SequenceTagger(
                hidden_size=settings.hidden_size,
                embeddings=embeddings,
                tag_dictionary=label_dict,
                tag_type=settings.task_name,
            )

        elif settings.task_name in ["sentiment", "class"]:

            if args.model_name in ["rnn", "lstm"]:
                document_embeddings = DocumentRNNEmbeddings(
                    embeddings=[FlairEmbeddings(args.saved_model)]
                )
            else:
                document_embeddings = NGMETransformerWordEmbeddings(
                    args.saved_model, vocab_file=args.saved_model + "/vocab.txt"
                )

            task_model = TextClassifier(
                document_embeddings=document_embeddings,
                label_dictionary=label_dict,
                label_type=settings.task_name,
            )
        else:
            print(f"Task {settings.task_name} not supported")
            exit()

        # Initialize trainer
        trainer = ModelTrainer(task_model, corpus)

        # Start training
        score = trainer.train(
            settings.save,
            learning_rate=settings.lr,
            mini_batch_size=settings.mini_batch_size,
            max_epochs=settings.max_epochs,
            # Bug with saving vocab file in saved model
            use_final_model_for_eval=True
            if args.model_name == "transformer"
            else False,
        )

        if isinstance(score, dict):
            score = score["test_score"]

        results[f"{settings.task_name}/f1-score"] = score

    if run:
        print("Upload scores to W&B run...", end="")
        for _task, score in results.items():
            run.summary[_task] = score

        run.update()
        print("Done")
    print(results)


if __name__ == "__main__":
    train_ds()
