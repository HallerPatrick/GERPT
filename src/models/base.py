from typing import Any, Optional


import pytorch_lightning as pl
from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
import torch

def human_readable_flops(num):
    for unit in [
        "",
        "KFLOPS",
        "MFLOPS",
        "GFLOPS",
        "TFLOPS",
        "PFLOPS",
        "EFLOPS",
        "ZFLOPS",
    ]:
        if abs(num) < 1000.0:
            return "%3.1f%s" % (num, unit)
        num /= 1000.0
    return "%.1f%s" % (num, "Yi")

class BasePLModel(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def register_flop_profiler(self, model):

        if torch.cuda.is_available():
            self.flop_profiler = FlopsProfiler(model)

    def on_train_batch_start(self, batch: Any, batch_idx: int, unused: int = 0) -> Optional[int]:


        if torch.cuda.is_available():
            if batch_idx % 2 ==  0:
                self.flop_profiler.start_profile()
        return super().on_train_batch_start(batch, batch_idx, unused)

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, unused: int = 0) -> None:

        if torch.cuda.is_available():
            if batch_idx % 2 ==  0:

                batch_size = batch["source"].size()
                seq_len = batch_size[1]
                batch_size = batch_size[2]

                duration = float(self.flop_profiler.get_total_duration())
                flops = int(self.flop_profiler.get_total_flops())

                total_flops = batch_size * seq_len * flops

                # human_flops = human_readable_flops(total_flops)
                
                self.log("train/flops_per_gpu_per_secs", total_flops / duration)
                self.flop_profiler.end_profile()


        return super().on_train_batch_end(outputs, batch, batch_idx, unused)
