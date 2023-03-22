import os
import shutil
from pathlib import Path

import torch

from src.args import parse_args
from src.process import process_tokenized_dataset


def main():
    args = parse_args()

    # torch.saved does not overwrite file for some reason
    if Path(args.saved_data).exists():
        print(f"Delete existing tokenized dataset: {args.saved_data}")
        try:
            shutil.rmtree(args.saved_data)
        except NotADirectoryError:
            os.remove(args.saved_data)

    dictionary = process_tokenized_dataset(
        args.saved_data,
        args.data,
        args.ngme,
        args.ngram,
        args.max_dict_size,
        args.min_frequency,
        args.cpus,
        args.is_forward,
        args.packed,
        args.reuse_dict,
        1000,
    )

    # Save dict, but dont overwrite
    if not args.reuse_dict:
        if not Path("dicts").exists():
            os.mkdir("dicts")

        # args.saved_dict is most likely a complete path, just take the file name
        dict_stem = Path(args.saved_dict).stem
        dict_file = str(Path("dicts") / (dict_stem + ".json"))

        dictionary.save_vocabulary(dict_file, args.ngram)


    print("Preprocessing done")
    print("=" * 80)
    print("Saving:")
    print(f"Tokenized dataset: {args.saved_data}")
    print(f"Dictionary object: {args.saved_dict}")
    print("=" * 80)

    # torch.save(tokenized_dataset, args.saved_data, pickle_protocol=4)
    torch.save(dictionary, args.saved_dict)


if __name__ == "__main__":
    main()
