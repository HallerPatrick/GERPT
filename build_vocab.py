"""
Script to build vocab files, instead of populating dictionaries
rom scratch during preprocessingself.

This only makes sense for explicit encoding.
"""



from src.args import argparse_build_dict
from src.dataset import load_dataset_from_source, populate_dense_dict
from src.dictionary import Dictionary


def main():

    args = argparse_build_dict()

    dataset = load_dataset_from_source(args.dataset)

    dictionary = Dictionary(
        args.ngrams, args.max_dict_size, "dense"
    )

    dictionary = populate_dense_dict(dictionary, args.ngrams, dataset["train"])

    new_ngram = 2
    new_dictionary = dictionary.unking(200, new_ngram)

    # new_dictionary.save_vocabulary("", new_ngram)



if __name__ == '__main__':
    main()
