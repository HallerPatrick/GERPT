from argparse import Namespace
from typing import Optional

from transformers import AutoModel, AutoConfig, AutoTokenizer

# from flair.embeddings import WordEmbeddings
from flair.embeddings.document import DocumentRNNEmbeddings

from src.args import argparse_flair_train, read_config
from src.models.flair_models import (
    load_corpus,
    patch_flair,
    NGMETransformerWordEmbeddings,
)

import wandb

import src.models.transformer

# from src.models.transformer.configuration_transformer import TransformerConfig
# from src.models.transformer.tokenization_transformer import NGMETokenizer


def train_ds(args: Optional[Namespace] = None, wandb_run_id: Optional[str] = None):

    if not args:
        args = read_config(argparse_flair_train().config)

    # Has to be called first, before importing flair modules

    if args.model_name == "rnn":
        patch_flair()

    from flair.embeddings import FlairEmbeddings, StackedEmbeddings
    from flair.models import SequenceTagger, TextClassifier
    from flair.trainers import ModelTrainer

    if wandb_run_id:
        args.wandb_run_id = wandb_run_id

        api = wandb.Api()
        run = api.run(args.wandb_run_id)
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

                task_model = SequenceTagger(
                    hidden_size=256,
                    embeddings=embeddings,
                    tag_dictionary=label_dict,
                    tag_type=settings.task_name,
                    use_crf=True,
                )

            else:
                # 4. initialize fine-tuneable transformer embeddings WITH document context
                embeddings = NGMETransformerWordEmbeddings(
                    "checkpoints/model.pt",
                    vocab_file="checkpoints/model.pt/vocab.json",
                    layers="-1",
                    subtoken_pooling="first",
                    fine_tune=True,
                    use_context=True,
                )

                # 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
                task_model = SequenceTagger(
                    hidden_size=256,
                    embeddings=embeddings,
                    tag_dictionary=label_dict,
                    tag_type="ner",
                    use_crf=False,
                    use_rnn=False,
                    reproject_embeddings=False,
                )

        elif settings.task_name in ["sentiment", "class"]:

            if args.model_name == "rnn":
                document_embeddings = DocumentRNNEmbeddings(
                    embeddings=[FlairEmbeddings(args.saved_model)]
                )
            else:
                # TODO make sequence tagging work
                document_embeddings = NGMETransformerWordEmbeddings(
                    "checkpoints/model.pt",
                    vocab_file="checkpoints/model.pt/vocab.json",
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
        )

        if isinstance(score, dict):
            score = score["test_score"]

        # Update run with results
        if run:
            run.summary[f"{settings.task_name}/f1-score"] = score
            run.update()

        results[f"{settings.task_name}/f1-score"] = score

    print(results)


if __name__ == "__main__":
    train_ds()
