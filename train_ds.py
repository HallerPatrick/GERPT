from argparse import Namespace
from typing import Optional

from flair import set_seed
from flair.embeddings import WordEmbeddings
from flair.embeddings.document import DocumentRNNEmbeddings
from torch import manual_seed

import wandb
from src.args import argparse_flair_train, read_config
from src.models.flair_models import (NGMETransformerWordEmbeddings,
                                     load_corpus, patch_flair_lstm, patch_flair_trans)


def train_ds(args: Optional[Namespace] = None, wandb_run_id: Optional[str] = None):
    """Downstream training procedure.

    Args:
        args(Namespace): Either directly read from config and command line or passed from pre-training
        wandb_run_id(Optional[str]): If passed from pre-training take existing wandb run id for logging
    """

    if not args:
        args = read_config(argparse_flair_train().config)

    assert args is not None

    # Seed everything
    set_seed(int(args.seed))
    manual_seed(int(args.seed))

    # Has to be called first, before importing flair modules
    if args.model_name == "rnn":
        patch_flair_lstm()
    else:
        patch_flair_trans()
    from flair.embeddings import FlairEmbeddings, StackedEmbeddings
    from flair.models import SequenceTagger, TextClassifier
    from flair.trainers import ModelTrainer

    # Try to log to existing wandb run

    if wandb.run is None:
        if wandb_run_id:
            pass
        elif args.wandb_run_id:
            pass
            # wandb.init(id=args.wandb_run_id, project="gerpt")

    results = {}

    charts = []

    for task_settings in args.downstream_tasks:

        settings = task_settings.task

        if not settings.use:
            print(f"Skipping fine-tune task {settings.task_name}")
            continue

        if wandb.run:
            wandb.run.config.update({settings.task_name: vars(settings)}, allow_val_change=True)

        corpus = load_corpus(settings.dataset, settings.base_path)

        label_dict = corpus.make_label_dictionary(label_type=settings.task_name)

        if settings.task_name in ["ner", "upos"]:

            if args.model_name == "rnn":
                
                if hasattr(args, "saved_model_backward"):
                    print("Using forward and backwards models")
                    embds = [
                        FlairEmbeddings(args.saved_model),
                        FlairEmbeddings(args.saved_model_backward),
                        WordEmbeddings("glove")
                    ]
                else:
                    embds = [
                        FlairEmbeddings(args.saved_model),
                        # WordEmbeddings("glove")
                    ]
                    
                embeddings = StackedEmbeddings(
                    embeddings=embds
                )
            else:

                embeddings = FlairEmbeddings(args.saved_model)
                # embeddings = NGMETransformerWordEmbeddings(
                #     args.saved_model,
                #     vocab_file=args.saved_model + "/vocab.txt",
                #     layers="all",
                #     subtoken_pooling="first",
                #     fine_tune=False,
                #     use_context=False,
                # )

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
                document_embeddings = DocumentRNNEmbeddings(
                    embeddings=[FlairEmbeddings(args.saved_model)]
                )
                # document_embeddings = NGMETransformerWordEmbeddings(
                #     args.saved_model, vocab_file=args.saved_model + "/vocab.txt"
                # )

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
            # if args.model_name == "transformer"
            # else False,
        )
        chart = None

        if isinstance(score, dict):
            test_score = score["test_score"]
            dev_scores = score["dev_score_history"]
            data = [[epoch+1, score] for epoch, score in enumerate(dev_scores)]
            table = wandb.Table(data=data, columns=["epoch", "f1-score(dev)"])
            fields = {"x": "epoch", "value": "f1-score(dev)"}
            chart = wandb.plot_table(
                vega_spec_name=f"{settings.task_name}/f1-score",
                data_table=table,
                fields=fields
            )
            charts.append(chart)

            results[f"{settings.task_name}/f1-score"] = test_score

    if wandb.run:
        print("Upload scores to W&B run...", end="")
        wandb.run.summary.update(results)
            
        if len(charts) > 0:
            for chart in charts:
                wandb.log({"downstream-training": chart})

        print("Done")
    print(results)


if __name__ == "__main__":
    train_ds()
