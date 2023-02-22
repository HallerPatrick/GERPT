import sys
import timeit
from math import log
from datetime import timedelta
from typing import Any, Callable, Optional, List

import pytorch_lightning as pl
import torch
from torch import Tensor

import pyarrow as pa
import numpy as np
from numba import jit

from datasets.features.features import _ArrayXDExtensionType, _is_zero_copy_only
from datasets.formatting.formatting import _is_array_with_nulls, _unnest, Formatter, BaseArrowExtractor

import torch.nn.functional as F

from prettytable import PrettyTable
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

strats = {
    "linear": lambda x: x,
    "log": lambda x: log(x+1),
    "exp": lambda x: x**2
}

def n_dist(n: int, strategy: str) -> list[float]:
    """dist of ngram weight is logarithmic"""
    ns = list(range(1, n+1))
    xs = list(map(strats[strategy], ns))
    result = list(map(lambda x: x / sum(xs), xs))
    return result


def split_range(i, ds_split, num_splits):
    split_len = len(ds_split)
    split_size = split_len // num_splits
    start, end = (i*split_size), ((i+1)*split_size)
    if end > split_len:
        end = split_len

    if i > 0:
        start += 1

    return start, end

def pack(integer_list):
    """Pack a list of integers. Maximum integer value is ~60000 and up to 4 in a list"""

    assert len(integer_list) <= 4

    packed_integer = 0
    for i, integer in enumerate(integer_list):
        packed_integer |= integer << (16 * i)
    return packed_integer

def pack_tensor(tensor: Tensor) -> Tensor:

    # Assume n-gram is first dimension
    assert len(tensor.shape) == 2
    
    return torch.tensor([pack(tensor[:, col_i]) for col_i in range(tensor.size(1))])
        
        
def unpack(packed_integer):
    """Unpack a list of integers. Maximum integer value is ~60000 and up to 4 in a list"""
    integer_list = []
    while packed_integer > 0:
        integer = packed_integer & (2**16 - 1)
        packed_integer >>= 16
        integer_list.append(integer)
    return integer_list

def unpack_tensor(tensor: Tensor) -> Tensor:
    assert len(tensor.shape) == 1
    
    unpacked_sequence = [unpack(i) for i in tensor]

    return torch.tensor(unpacked_sequence, dtype=torch.int64).t().to(tensor.device)

def unpack_batched_tensor(tensor: Tensor) -> Tensor:

    assert len(tensor.shape) == 2, f"But dim is {tensor.shape}"

    # Expect batch dimension second
    ts = [unpack_tensor(tensor[:, batch_i]).unsqueeze(-1) for batch_i in range(tensor.size(1))]

    return torch.cat(ts, dim=-1).to(tensor.device)




class TimePerEpochCallback(Callback):
    def on_train_epoch_start(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.start = timeit.default_timer()
        return super().on_train_epoch_start(trainer, pl_module)

    def on_train_epoch_end(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        end = timeit.default_timer()
        trainer.logger.log_metrics(
            {"train/secs_per_epoch": timedelta(seconds=end - self.start).seconds}
        )
        return super().on_train_epoch_end(trainer, pl_module)


class ModelCheckpointCallback(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> Optional[dict]:
        self.save_path = f"{self.dirpath}/{self.filename}.ckpt"
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)


def get_encoder_params(model):
    for name, parameter in model.named_parameters():
        if name.startswith("encoder"):
            return parameter.numel()


def display_text(t, dictionary, ngram):
    for a in t:
        print(repr(dictionary.ngram2idx2word[ngram][a.item()]), end="")
    print()



def display_prediction(prediction, dictionary):
    prediction = F.softmax(prediction.view(-1), dim=0)
    preds = []
    for i, pred in enumerate(prediction):
        preds.append((i, pred.item()))

    preds = sorted(preds, key=lambda x: x[1], reverse=True)

    for p in preds:
        i, pred = p
        print("{:9}: {:.15f},".format(repr(dictionary.idx2word[i]), pred))


def calcualate_transformer_hidden_size(d: int, e: int, l: int, h: int, hid: int, total_size: int) -> int:
    """

    Args
    ----
    d: dict size
    e: embedding size
    h: heads
    l: layers
    hid: hidden size

    """

    # Lazy load wildcard, takes some time
    from sympy import Symbol
    from sympy.solvers import solve



    hid = Symbol("hid")

    result = solve(
        ( d * e + e ) # encoder
        + l * (
            (e * (3 * e) + (3 * e)) #atnn_in
            + (e * e + e) #atnn_out
            + (e * hid + hid) #layer1
            + (e * hid + e) #layer2
            + (2 * (2 * e)) #norm
        )
        + (d * e + e) # decoder
        - total_size,
        hid

    )

    if isinstance(result, list):
        return result[0].evalf()

    return result.evalf()


def lstm_size(e, h):
    c = 4
    lstm_size = (
        e * h + h * h + h + h
    ) * 4

    lstm_size = 4 * ( h * (e + h) + h + h )
    return lstm_size


def calculate_lstm_hidden_size(d: int, e: int, c: int, l: int, total_size: int, h):
    """

    Args
    ----
    d: dict size
    e: embedding size
    c: FNNN Units (LSTM=4)
    l: layers
    h: hidden size

    """


    from sympy import Symbol
    from sympy.solvers import solve


    encoder_size = d * e + e
    lstm_size = (
            (c * e * h)
            + (c * h * h)
            + (c * h)
            + (c * h)
            + (l - 1) * ((c * h * h) + (c * h * h) + (c * h) + (c * h))
    )
    decoder_size = d * h + d
    print(f"Encoder Size: {encoder_size}")
    print(f"LSTM size: {lstm_size}")
    print(f"Decoder Size: {decoder_size}")
    print(f"Actual (calculated) model size: {encoder_size + lstm_size + decoder_size}")

    h = Symbol("h")

    result = solve(
        (d * e + e)
        + (
                (c * e * h)
                + (c * h * h)
                + (c * h)
                + (c * h)
                + (l - 1) * ((c * h * h) + (c * h * h) + (c * h) + (c * h))
        )
        + (d * h + d)
        - total_size,
        h,
    )

    if isinstance(result, list):
        return result[0].evalf()

    return result.evalf()


class DummyLogger:
    """Dummy logger for internal use.
    It is useful if we want to disable user's logger for a feature, but still ensure that user code can run
    """

    def __init__(self) -> None:
        super().__init__()
        # self._experiment = DummyExperiment()

    def log_metrics(self, *args: Any, **kwargs: Any) -> None:
        pass

    def log_hyperparams(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def name(self) -> str:
        """Return the experiment name."""
        return ""

    @property
    def version(self) -> str:
        """Return the experiment version."""
        return ""

    def __getitem__(self, idx: int) -> "DummyLogger":
        return self

    def __getattr__(self, name: str) -> Callable:
        """Allows the DummyLogger to be called with arbitrary methods, to avoid AttributeErrors."""

        def method(*args: Any, **kwargs: Any) -> None:
            return None

        return method


def collect_token_metrics(dictionary, ngram: int):
    dset_metrics = {}

    dset_metrics[f"total_tokens"] = sum(dictionary.total_n_tokens.values()) + sum(
        dictionary.unk_n_tokens.values()
    )
    for n in range(1, ngram + 1):
        dset_metrics[f"total_{n}_gram_tokens"] = dictionary.total_n_tokens[n]
        dset_metrics[f"total_{n}_gram_unk_tokens"] = dictionary.unk_n_tokens[n]

    return dset_metrics


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def concat_dataset(rows: List[List[List[int]]]):
    # chunksize = calc_chunksize(rows, num_workers)
    # chunksize = 10000
    # print(f"Chunksize: {chunksize}")

    # with Pool(num_workers) as pool:
    # sublists = process_map(np_array, rows, max_workers=num_workers, chunksize=chunksize)
    
    return np.concatenate((rows), axis=1, dtype=np.int16)

@jit(parallel=True, nopython=True)
def numba_concat_dataset(rows: List[List[List[int]]]):
    return np.concatenate(rows, axis=1) #, dtype=np.int16)



def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


class StackedNumpyArrowExtractor(BaseArrowExtractor[dict, np.ndarray, dict]):
    def __init__(self, **np_array_kwargs):
        self.np_array_kwargs = np_array_kwargs

    def extract_row(self, pa_table: pa.Table) -> dict:
        return _unnest(self.extract_batch(pa_table))

    def extract_column(self, pa_table: pa.Table) -> np.ndarray:
        return self._arrow_array_to_numpy(pa_table[pa_table.column_names[0]])

    def extract_batch(self, pa_table: pa.Table) -> dict:
        return {col: self._arrow_array_to_numpy(pa_table[col]) for col in pa_table.column_names}

    def _arrow_array_to_numpy(self, pa_array: pa.Array) -> np.ndarray:
        if isinstance(pa_array, pa.ChunkedArray):
            if isinstance(pa_array.type, _ArrayXDExtensionType):
                # don't call to_pylist() to preserve dtype of the fixed-size array
                zero_copy_only = _is_zero_copy_only(pa_array.type.storage_dtype, unnest=True)
                if pa_array.type.shape[0] is None:
                    array: List = [
                        row
                        for chunk in pa_array.chunks
                        for row in chunk.to_list_of_numpy(zero_copy_only=zero_copy_only)
                    ]
                else:
                    array: List = [
                        row for chunk in pa_array.chunks for row in chunk.to_numpy(zero_copy_only=zero_copy_only)
                    ]
            else:
                zero_copy_only = _is_zero_copy_only(pa_array.type) and all(
                    not _is_array_with_nulls(chunk) for chunk in pa_array.chunks
                )
                array: List = [
                    row for chunk in pa_array.chunks for row in chunk.to_numpy(zero_copy_only=zero_copy_only)
                ]
        else:
            if isinstance(pa_array.type, _ArrayXDExtensionType):
                # don't call to_pylist() to preserve dtype of the fixed-size array
                zero_copy_only = _is_zero_copy_only(pa_array.type.storage_dtype, unnest=True)
                if pa_array.type.shape[0] is None:
                    array: List = pa_array.to_list_of_numpy(zero_copy_only=zero_copy_only)
                else:
                    array: List = pa_array.to_numpy(zero_copy_only=zero_copy_only)
            else:
                zero_copy_only = _is_zero_copy_only(pa_array.type) and not _is_array_with_nulls(pa_array)
                array: List = pa_array.to_numpy(zero_copy_only=zero_copy_only).tolist()
        if len(array) > 0:
            if any(
                (isinstance(x, np.ndarray) and (x.dtype == object or x.shape != array[0].shape))
                or (isinstance(x, float) and np.isnan(x))
                for x in array
            ):
                if isinstance(array[0], np.ndarray):
                    array = array[0]
                
                result = np.stack(np.array(array, copy=False, **{**self.np_array_kwargs, "dtype": object}))
                return [result]
        return np.array(array, copy=False, **self.np_array_kwargs)


class StackedNumpyFormatter(Formatter[dict, np.ndarray, dict]):

    numpy_arrow_extractor = StackedNumpyArrowExtractor

    def __init__(self, features=None, decoded=True, **np_array_kwargs):
        super().__init__(features=features, decoded=decoded)
        self.np_array_kwargs = np_array_kwargs

    def format_row(self, pa_table: pa.Table) -> dict:
        row = self.numpy_arrow_extractor(**self.np_array_kwargs).extract_row(pa_table)
        if self.decoded:
            row = self.python_features_decoder.decode_row(row)
        return row

    def format_column(self, pa_table: pa.Table) -> np.ndarray:
        column = self.numpy_arrow_extractor(**self.np_array_kwargs).extract_column(pa_table)
        if self.decoded:
            column = self.python_features_decoder.decode_column(column, pa_table.column_names[0])
        return column

    def format_batch(self, pa_table: pa.Table) -> dict:
        batch = self.numpy_arrow_extractor(**self.np_array_kwargs).extract_batch(pa_table)
        if self.decoded:
            batch = self.python_features_decoder.decode_batch(batch)
        return batch

def set_stacked_numpy_formatter():
    from datasets.formatting import _register_formatter

    _register_formatter(StackedNumpyFormatter, "stacked-numpy", aliases=["snp"])

