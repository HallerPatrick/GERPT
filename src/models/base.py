from typing import Any, Optional


import pytorch_lightning as pl
from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
import torch


class BasePLModel(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def register_flop_profiler(self, model):

        if torch.cuda.is_available():
            self.flop_profiler = FlopsProfiler(model)

    def on_train_batch_start(self, batch: Any, batch_idx: int, unused: int = 0) -> Optional[int]:

        if torch.cuda.is_available():
            if batch_idx == 0:
                self.flop_profiler.start_profile()
        return super().on_train_batch_start(batch, batch_idx, unused)

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, unused: int = 0) -> None:

        if torch.cuda.is_available():
            if batch_idx == 0:
                print(f"Total duration of forward pass: {self.flop_profiler.get_total_duration()}")
                print(f"Total FLOPS: {self.flop_profiler.get_total_flops()}")
                flops = self.flop_profiler.get_total_flops(as_string=True)
                params = self.flop_profiler.get_total_params(as_string=True)
                self.flop_profiler.print_model_profile(profile_step=batch_idx)
                self.flop_profiler.end_profile()


        return super().on_train_batch_end(outputs, batch, batch_idx, unused)
