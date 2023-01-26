import numpy as np

local_dataset_mapper = {
    "hp": {
        "train": "data/hp/train.txt",
        "test": "data/hp/test.txt",
        "validation": "data/hp/valid.txt",
    },
    "cash": {
        "train": "data/cash/train.txt",
        "test": "data/cash/test.txt",
        "validation": "data/cash/valid.txt",
    },
    "wikitext-2": {
        "train": "data/wikitext-2/train.txt",
        "test": "data/wikitext-2/test.txt",
        "validation": "data/wikitext-2/valid.txt",
    },
    # HC
    "cc100_german": {
        "train": "home/tmp/halerpat/data/train.txt",
        "test": "home/tmp/halerpat/data/test.txt",
        "validation": "home/tmp/halerpat/data/valid.txt",
    },
    "wikipedia_en": {"args": ["wikipedia", "20220301.en"]},
    "wikipedia_de": {"args": ["wikipedia", "20220301.de"]},
    "obw_news": {"train": "data/obw_news/train.txt"},
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

    assert len(text.shape) == 2

    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    nbatch = text.shape[1] // (batch_size * bptt)
    text = text[:, : nbatch * batch_size * bptt]
    text = text.reshape((text.shape[0], batch_size, -1)).transpose((0, 2, 1))

    return text, nbatch
