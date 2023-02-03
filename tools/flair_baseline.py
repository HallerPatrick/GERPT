from flair.embeddings import FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from src.models.flair_models import load_corpus


if __name__ == '__main__':

    hidden_size = 256

    corpus = load_corpus("conll_03", "data")

    label_dict = corpus.make_label_dictionary(label_type="ner")

    embeddings = FlairEmbeddings("news-forward")

    task_model = SequenceTagger(
        hidden_size=hidden_size,
        embeddings=embeddings,
        tag_dictionary=label_dict,
        tag_type="ner",
    )


    trainer = ModelTrainer(task_model, corpus)

    # Start training
    score = trainer.train(
        "resources/ner",
        learning_rate=0.1,
        mini_batch_size=32,
        max_epochs=150,
        # Bug with saving vocab file in saved model
        use_final_model_for_eval=True
        # if args.model_name == "transformer"
        # else False,
    )
