import math
from typing import Optional

import pytorch_lightning as pl
import torch
from rich import print
from rich.panel import Panel

import wandb
from src.dictionary import Dictionary
from src.models.ngme import soft_n_hot

# from src.models.transformer import TransformerConfig, TransformerTransformer

DEBUG = False

# class TransformerLightningModule(pl.LightningModule):
#     def __init__(
#         self, config: TransformerConfig, dictionary: Optional[Dictionary] = None
#     ):
#
#         super(TransformerLightningModule, self).__init__()
#
#         self.model = TransformerTransformer(config)
#         self.config = config
#
#         self.epoch = 0
#         self.dictionary = dictionary
#
#     def _step(self, batch):
#         # source, target = batch[0], batch[1]
#
#         source, target = batch[0].permute((1, 2, 0)), batch[1].permute((1, 2, 0))
#
#         if DEBUG:
#             print("Source:")
#             self.dictionary.print_batch(source)
#             print("Target:")
#             self.dictionary.print_batch(target)
#             input("continue")
#
#         output = self.model.forward(source)
#         output = output.view(-1, self.model.ntoken)
#         target = soft_n_hot(
#             target, self.model.ntoken, self.model.weighted_labels
#         )
#         target = target.view(-1, self.model.ntoken)
#         return self.model.criterion(output, target)
#
#     def training_step(self, batch, batch_idx):
#         loss = self._step(batch)
#         self.log("train/loss", loss)
#         self.log("train/ppl", math.exp(loss), prog_bar=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         loss = self._step(batch)
#         self.log("val/loss", loss)
#         self.log("val/ppl", math.exp(loss))
#
#     def test_step(self, batch, batch_idx):
#         loss = self._step(batch)
#         self.log("test/loss", loss)
#         self.log("test/ppl", math.exp(loss))
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.SGD(self.parameters(), lr=5.0)
#         lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             "min",
#             factor=0.25,
#             verbose=True,
#             min_lr=1.25,
#             threshold=1e-6,
#             patience=10000,
#             threshold_mode="abs",
#         )
#         return [optimizer], [
#             {"scheduler": lr_scheduler, "monitor": "train/loss", "interval": "step"}
#         ]
#
#     def training_epoch_end(self, outputs) -> None:
#
#         self.epoch += 1
#
#         if self.config.generate and self.dictionary:
#             _, output = self.generate_text()
#             print(Panel(output, title="[green]Generated text"))
#             self.train()
#
#     def generate_text(self) -> str:
#
#         assert self.dictionary is not None
#
#         ntokens = len(self.dictionary)
#
#         inp = torch.randint(
#             ntokens, (self.config.ngrams, 1, 1), dtype=torch.long, device=self.device
#         )
#         idx = inp[0][0].detach()
#         generated_output = self.dictionary.get_item_for_index(idx.item())
#         printed_output = self.dictionary.get_item_for_index(idx.item())
#
#         with torch.no_grad():
#             self.eval()
#             for i in range(self.config.chars):
#
#                 output = self.model(inp, False)
#
#                 # Only use the generated ngrams
#                 output = output[-1]
#
#                 if self.config.temperature == 0.0:
#                     # output = F.softmax(output, dim=0).cpu()
#                     # Just get highest confidence
#                     ngram_idx = torch.argmax(output)
#                 else:
#                     # output = F.log_softmax(output, dim=0)
#
#                     word_weights = (
#                         output[-1].squeeze().div(self.config.temperature).exp().cpu()
#                     )
#                     # multinomial over all tokens
#                     ngram_idx = torch.multinomial(word_weights, 1)[0]
#
#                 # Get ngram word
#                 word = self.dictionary.get_item_for_index(ngram_idx.item())
#
#                 # Append to generated sequence
#                 printed_output = printed_output + word + "·"
#                 generated_output = generated_output + word
#
#                 # Use last 200 chars as sequence for new input
#                 inp = (
#                     self.dictionary.tokenize_line(
#                         list(generated_output[-200:]), id_type=torch.int64, return_tensor="pt"
#                     )["source"]
#                     .unsqueeze(dim=2)
#                     .to(self.device)
#                 )
#                 # print(inp.size())
#
#             self.train()
#
#         self.logger.log_text(
#             "samples",
#             columns=["epoch", "temperatue", "text"],
#             data=[[self.epoch, self.config.temperature, printed_output]],
#         )
#
#         return generated_output, printed_output
