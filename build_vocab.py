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

    dataset, write_strategy = load_dataset_from_source(args.dataset)

    dictionary, dataset = Dictionary.build_from_dataset(
        dataset, args.ngrams, args.max_dict_size, 7000
    )

    dictionary = dictionary.unking(ngrams=3, min_frequency=1000, remove_whitespace_tokens=True)

    dictionary.save_vocabulary(args.output, args.ngrams)
    print(len(dictionary))


if __name__ == '__main__':
    main()
