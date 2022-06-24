from datetime import timedelta
from math import sqrt
import timeit

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
        (
            (c * e * h)
            + (c * h * h)
            + (c * h)
            + (c * h)
            + (l - 1) * ((c * h * h) + (c * h * h) + (c * h) + (c * h))
        )
        - total_size,
        h,
    )

    if isinstance(result, list):
        return result[0].evalf()

    return result.evalf()
