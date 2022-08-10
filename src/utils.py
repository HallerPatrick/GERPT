from datetime import timedelta
from math import sqrt
from typing import Any, Callable, Optional
import timeit
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback

from sympy.solvers import solve
from sympy import *


class TimePerEpochCallback(Callback):
    def on_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.start = timeit.default_timer()
        return super().on_epoch_start(trainer, pl_module)

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        end = timeit.default_timer()
        trainer.logger.log_metrics(
            {"train/secs_per_epoch": timedelta(seconds=end - self.start).seconds}
        )
        return super().on_epoch_end(trainer, pl_module)

class ModelCheckpointCallback(ModelCheckpoint):

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> Optional[dict]:
        self.save_path = f"{self.dirpath}/{self.filename}.ckpt"
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)

def get_encoder_params(model):
    for name, parameter in model.named_parameters():
        if name.startswith("encoder"):
            return parameter.numel()


def display_text(dictionary, t):
    for a in t:
        print(repr(dictionary.idx2word[a.item()]), end="")
    print()


def display_input_n_gram_sequences(input, dictionary):
    for i in range(input.size()[0]):
        print(f"{i+1}-gram")
        display_text(dictionary, input[i])


def display_prediction(prediction, dictionary):
    prediction = F.softmax(prediction.view(-1), dim=0)
    preds = []
    for i, pred in enumerate(prediction):
        preds.append((i, pred.item()))

    preds = sorted(preds, key=lambda x: x[1], reverse=True)

    for p in preds:
        i, pred = p
        print("{:9}: {:.15f},".format(repr(dictionary.idx2word[i]), pred))


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
        (d * e + e) + 
        (
            (c * e * h)
            + (c * h * h)
            + (c * h)
            + (c * h)
            + (l - 1) * ((c * h * h) + (c * h * h) + (c * h) + (c * h))
        ) + 
        (d * h + d)
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

    # @property
    # def experiment(self) -> DummyExperiment:
    #     """Return the experiment object associated with this logger."""
    #     return self._experiment

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
        # enables self.logger[0].experiment.add_image(...)
        return self

    # def __iter__(self) -> Generator[None, None, None]:
    #     # if DummyLogger is substituting a logger collection, pretend it is empty
    #     yield from ()

    def __getattr__(self, name: str) -> Callable:
        """Allows the DummyLogger to be called with arbitrary methods, to avoid AttributeErrors."""

        def method(*args: Any, **kwargs: Any) -> None:
            return None

        return method
