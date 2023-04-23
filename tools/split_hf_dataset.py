import pathlib

import datasets
from tqdm import tqdm


SPLIT_THRESHOLD = 100


def main():

    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")

    target_dir = pathlib.Path("data/wiki_split/train")

    if not target_dir.exists():
        target_dir.mkdir(parents=True)

    train_split = dataset["train"]

    num_splits = len(train_split) // SPLIT_THRESHOLD
    len_zeros = len(str(num_splits))



    for j, i in tqdm(enumerate(range(0, len(train_split), SPLIT_THRESHOLD))):
        if i+SPLIT_THRESHOLD <= len(train_split):
            split = train_split[i: i+SPLIT_THRESHOLD]
            split_text = "\n".join(split["text"])

            split_path = target_dir / f"split-{j+1:0{len_zeros}}-of-{num_splits}"

            with open(split_path, "w") as f:
                f.write(split_text)

    print(f"Num of splits: {num_splits}")


if __name__ == '__main__':
    main()
