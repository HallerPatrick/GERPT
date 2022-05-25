
from src.args import argparse_flair_train, read_config
from src.models.flair_models import load_corpus, patch_flair

import wandb

from flair.embeddings import FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

args = read_config(argparse_flair_train().config)

# Has to be called first, before importing flair modules
patch_flair(args.model_name)

api = wandb.Api()
run = api.run(args.wandb_run_id)

for task_settings in args.downstream_tasks:

    settings = task_settings.task

    run.config.update({settings.task_name: vars(settings)})
    run.update()

    corpus = load_corpus(settings.dataset, settings.base_path)
    
    label_dict = corpus.make_label_dictionary(label_type=settings.task_name)

    embedding_types = [FlairEmbeddings(args.saved_model)]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=label_dict,
        tag_type=settings.task_name,
        use_crf=True,
    )

    # Initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    # Start training
    score = trainer.train(
        settings.save,
        learning_rate=settings.lr,
        mini_batch_size=settings.mini_batch_size,
        max_epochs=settings.max_epochs,
    )
        
    # Update run with results
    run.summary[settings.task_name] = score
    run.summary.update()
    


