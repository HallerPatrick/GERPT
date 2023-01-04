import timeit
from datetime import timedelta
from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn.functional as F
from prettytable import PrettyTable
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

# Lazy load wildcard, takes some time
from sympy import *
from sympy.solvers import solve

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

    assert len(tensor.shape) == 2

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
