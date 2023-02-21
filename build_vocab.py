"""
Script to build vocab files, instead of populating dictionaries
rom scratch during preprocessingself.

This only makes sense for explicit encoding.
"""

from src.args import argparse_build_dict
from src.process import load_dataset_from_source
from src.dictionary import Dictionary

def main():

    args = argparse_build_dict()

    dataset = load_dataset_from_source(args.dataset)

    dictionary, dataset = Dictionary.build_from_dataset(
        dataset, args.ngrams, args.max_dict_size, args.ngme, packed=False
    )

    dictionary.save_vocabulary(args.output, args.ngrams)


if __name__ == '__main__':
    main()
