import torch
import numpy as np

babylm_files = [
    "aochildes",
    "bnc_spoken",
    "cbt",
    "children_stories",
    "gutenberg",
    "open_subtitles",
    "qed",
    "simple_wikipedia",
    "switchboard",
    "wikipedia",
]


def baby_lm_train(size_path):
    return [f"data/babylm_data/{size_path}/{file}.train" for file in babylm_files]


def baby_lm_dev():
    return [f"data/babylm_data/babylm_dev/{file}.dev" for file in babylm_files]


def baby_lm_test():
    return [f"data/babylm_data/babylm_test/{file}.test" for file in babylm_files]


# Mapping of local datasets to their paths
local_dataset_mapper = {
    "hp": {
        "strategy": "default",
        "splits": {
            "train": "data/hp/train.txt",
            "test": "data/hp/test.txt",
            "valid": "data/hp/valid.txt",
        },
    },
    "cash": {
        "strategy": "default",
        "splits": {
            "train": "data/cash/train.txt",
            "test": "data/cash/test.txt",
            "valid": "data/cash/valid.txt",
        },
    },
    "cash_splits": {
        "strategy": "split",
        "splits": {
            "train": "data/cash_splits/train-{001..003}.txt",
        },
    },
    "obw": {
        "strategy": "split",
        "splits": {"train": "data/obw/train/news.en-{00001..00099}-of-00100"},
    },
    "c4": {
        "strategy": "split",
        "splits": {"train": "data/c4/train/split-{001..364}-of-364"},
    },
    "obw_news": {
        "train": "data/obw_news/train.txt",
        "test": "data/obw_news/test.txt",
        "valid": "data/obw_news/valid.txt",
    },
    "babylm10M_hf": {
        "train": baby_lm_train("babylm_10M"),
        "test": baby_lm_test(),
        # For the pipeline, which expects a validation set
        "validation": baby_lm_dev(),
    },
    "babylm10M": {
        "train": baby_lm_train("babylm_10M"),
        "test": baby_lm_test(),
        # For the pipeline, which expects a validation set
        "validation": baby_lm_dev(),
    },
    "babylm100M": {
        "train": baby_lm_train("babylm_100M"),
        "test": baby_lm_test(),
        # For the pipeline, which expects a validation set
        "validation": baby_lm_dev(),
    },
    # # HC
    # "cc100_german": {
    #     "train": "home/tmp/halerpat/data/train.txt",
    #     "test": "home/tmp/halerpat/data/test.txt",
    #     "valid": "home/tmp/halerpat/data/valid.txt",
    # },
    # Hugginface
    "wikipedia_en": {"args": ["wikipedia", "20220301.en"]},
    "wikipedia_de": {"args": ["wikipedia", "20220301.de"]},
}


def batchify(text: np.ndarray, batch_size: int, bptt: int):
    """Splits text into batches of size batch_size and bptt.
    Parameters
    ----------
    text: str
        Text to be batchified
    batch_size: int
        Number of batches
    bptt: int
        Number of tokens per batch
    """

    if text.shape[0] == 0:
        return torch.tensor([]), 0

    assert (
        len(text.shape) == 2
    ), f"Array should be 2-dimension not of shape: {text.shape}"

    text: torch.Tensor = torch.from_numpy(text)

    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    nbatch = text.size(1) // (batch_size * bptt)
    text = text[:, : nbatch * batch_size * bptt]

    # text = text.reshape((text.shape[0], batch_size, -1)).transpose((0, 2, 1))
    text = (
        text.view((text.size(0), batch_size, -1))
        .permute((0, 2, 1))
        .contiguous()
        .to(torch.int64)
    )

    # text: [ngram, seq, batch_size]
    return text, nbatch
