import shutil
import os

from pathlib import Path

import torch

from src.args import parse_args
from src.dataset import load_tokenized_dataset


def main():

    args = parse_args()

    tokenized_dataset, dictionary = load_tokenized_dataset(
        args.ngme,
        args.ngram,
        args.model,
        args.max_dict_size,
        args.unk_threshold,
        args.fallback,
        args.cpus,
        args.is_forward,
        args.packed,
        args.reuse_dict,
        *args.data.split("/")
    )
    
    # torch.saved does not overwrite file for some reason
    if Path(args.saved_data).exists():
        print(f"Delete existing tokenized dataset: {args.saved_data}")
        try:
            shutil.rmtree(args.saved_data)
        except NotADirectoryError:
            os.remove(args.saved_data)
    
    # Save dict, but dont overwrite
    if not args.reuse_dict:

        if not Path("dicts").exists():
            os.mkdir("dicts")

        # args.saved_dict is most likely a complete path, just take the file name
        dict_stem = Path(args.saved_dict).stem

        dictionary.save_vocabulary("dicts/" + dict_stem + ".dict", args.ngram)

    torch.save(tokenized_dataset, args.saved_data, pickle_protocol=4)
    torch.save(dictionary, args.saved_dict)

    print(dictionary.ngram2idx2word)


if __name__ == "__main__":
    main()
