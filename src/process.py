"""Process the dataset and build the dictionary"""

from typing import Iterable, Optional, Tuple, Union

import braceexpand
from datasets import load_dataset as ld
from datasets.dataset_dict import DatasetDict
import torch
from tqdm import tqdm

from src.data import local_dataset_mapper
from src.dictionary import Dictionary
from src.processor import Processor

def dataset_iterator(paths):
    for path in paths:
        yield ld("text", data_files={"train": path})


def load_dataset_from_source(ds_path: str) -> Tuple[Union[Iterable[DatasetDict], DatasetDict], str]:
    """We using the HF dataset path convenientions.
    Usually:
    <dataset>/<subset>

    For loading local datset, use:
    text/<dataset-path>

    We map <dataset-path> to dict of target files.
    """

    prefix, subset = ds_path.split("/")

    write_strategy = None

    # Check if we have a local config for local dataset
    if prefix == "text" and subset in local_dataset_mapper:
        if subset in ["obw", "cash_splits"]:
            write_strategy = local_dataset_mapper[subset]["strategy"]
            train_paths = list(
                braceexpand.braceexpand(local_dataset_mapper[subset]["splits"]["train"])
            )
            return dataset_iterator(train_paths), write_strategy

        write_strategy = local_dataset_mapper[subset]["strategy"]
        dataset = ld("text", data_files=local_dataset_mapper[subset]["splits"])

    # Special case
    elif prefix.startswith("wikipedia"):
        write_strategy = local_dataset_mapper[subset]["strategy"]
        dataset = ld(*local_dataset_mapper[prefix]["args"])
    else:
        write_strategy = "default"
        # Load the datasets from huggingface
        dataset = ld(*ds_path.split("/"))

    if "validation" in dataset:
        valid_split = dataset["validation"]
        del dataset["validation"]
        dataset["valid"] = valid_split

    assert write_strategy is not None, "Write strategy not found"

    return dataset, write_strategy


def process_tokenized_dataset(
    target_path,
    dataset_path: str,
    ngme: str,
    ngram: int,
    max_dict_size: int,
    num_proc: int,
    is_forward: bool,
    packed: bool,
    dict_file_name: Optional[str] = None,
    rows_threshold: int = 50_000_000,
) -> Dictionary:
    """🤗"""

    dataset, write_strategy = load_dataset_from_source(dataset_path)

    if dict_file_name:
        print("Loading dictionary...", end="")
        dictionary = Dictionary.load_from_file(dict_file_name)
        print("Done")
    else:
        print("Building dictionary...", end="")
        # This might take some while...
        dictionary, dataset = Dictionary.build_from_dataset(
            dataset, ngram, max_dict_size, ngme, packed
        )
        print("Done")

    processor = Processor.from_strategy(write_strategy)(
        target_path,
        dictionary,
        ngram,
        is_forward,
        num_proc,
        rows_threshold=rows_threshold,
    )

    print("Using processor: ", processor.__class__.__name__)

    # Main processing with tokenization
    processor.run(dataset)

    return dictionary
