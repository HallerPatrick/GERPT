import torch

from src.args import parse_args
from src.dataset import load_tokenized_dataset


def main():

    args = parse_args()

    tokenized_dataset, dictionary = load_tokenized_dataset(
        args.batch_size,
        args.bptt,
        args.ngram,
        args.max_dict_size,
        args.unk_threshold,
        args.fallback,
        args.cpus,
        args.is_forward,
        False,
        *args.data.split("/")
        # cache_dir="/home/tmp/halerpat/datasets"
    )

    tokenized_dataset.save_to_disk(args.saved_data)
    torch.save(dictionary, args.saved_dict)

    print(dictionary.ngram2idx2word)


if __name__ == "__main__":
    main()
