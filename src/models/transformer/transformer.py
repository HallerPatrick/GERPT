import math

import pytorch_lightning as pl
import torch
# from rich import print
# from rich.panel import Panel

from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer

from src.models.ngme import soft_n_hot
from src.models.transformer import TransformerConfig, TransformerTransformer

class TransformerLightningModule(pl.LightningModule):
    def __init__(
        self,
        config: TransformerConfig,
    ):
        super(TransformerLightningModule, self).__init__()
        self.model = TransformerTransformer(config)
        print(self.model.config)

        self.epoch = 0

    def training_step(self, batch, batch_idx):
        output = self.model.forward(batch["source"])
        output = output.view(-1, self.model.ntoken)
        target = soft_n_hot(batch["target"], self.model.ntoken, self.model.weighted_labels)
        target = target.view(-1, self.model.ntoken)
        loss = self.model.criterion(output, target)
        self.log("train/loss", loss)
        self.log("train/ppl", math.exp(loss), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model.forward(batch["source"])
        output = output.view(-1, self.model.ntoken)
        target = soft_n_hot(batch["target"], self.model.ntoken, self.model.weighted_labels)
        target = target.view(-1, self.model.ntoken)
        loss = self.model.criterion(output, target)
        self.log("val/loss", loss)
        self.log("val/ppl", math.exp(loss))

    def test_step(self, batch, batch_idx):
        output = self.model.forward(batch["source"])
        output = output.view(-1, self.model.ntoken)
        target = soft_n_hot(batch["target"], self.model.ntoken, self.model.weighted_labels)
        target = target.view(-1, self.model.ntoken)
        loss = self.model.criterion(output, target)
        self.log("test/loss", loss)
        self.log("test/ppl", math.exp(loss))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return optimizer  # ], [lr_scheduler]



    # def training_epoch_end(self, outputs) -> None:
    #
    #     self.epoch += 1
    #
    #     if self.model.generate:
    #         print(Panel(self.generate_text(), title="[green]Generated text"))
    #         self.train()
    #
    # def generate_text(self) -> str:
    #
    #     if self.model.ngrams > 2:
    #         unk_idxs = set(
    #             [
    #                 self.model.dictionary.word2idx[f"<{i}-UNK>"]
    #                 for i in range(3, self.model.ngrams + 1)
    #             ]
    #         )
    #         token_idxs = set(self.model.dictionary.word2idx.values())
    #         token_idxs = torch.tensor(
    #             list(token_idxs.difference(unk_idxs)), dtype=torch.int64
    #         )
    #
    #     ntokens = len(self.model.dictionary)
    #
    #     inp = torch.randint(
    #         ntokens, (self.model.ngrams, 1, 1), dtype=torch.long, device=self.device
    #     )
    #     generated_output = self.model.dictionary.idx2word[inp[0][0].item()]
    #
    #     with torch.no_grad():
    #         self.eval()
    #         for i in range(self.model.chars):
    #
    #             output = self.model(inp, False)
    #
    #             # Only use the generated ngrams
    #             output = output[-1]
    #
    #             if self.model.temperature == 0.0:
    #                 # output = F.softmax(output, dim=0).cpu()
    #                 # Just get highest confidence
    #                 ngram_idx = torch.argmax(output)
    #             else:
    #                 # output = F.log_softmax(output, dim=0)
    #
    #                 # Remove all UNK tokens for ngram > 2
    #                 if self.model.ngrams > 2:
    #                     output = torch.index_select(output, 0, token_idxs).cpu()
    #
    #                 word_weights = (
    #                     output[-1].squeeze().div(self.model.temperature).exp().cpu()
    #                 )
    #                 # multinomial over all tokens
    #                 ngram_idx = torch.multinomial(word_weights, 1)[0]
    #
    #             # Get ngram word
    #             word = self.model.dictionary.idx2word[ngram_idx]
    #
    #             # Append to generated sequence
    #             generated_output = generated_output + word
    #
    #             # Use last 200 chars as sequence for new input
    #             inp = (
    #                 self.model.dictionary.tokenize_line(
    #                     generated_output[-200:],
    #                 )["source"]
    #                 .unsqueeze(dim=2)
    #                 .to(self.device)
    #             )
    #             # print(inp.size())
    #
    #         self.train()
    #
    #     self.logger.log_text(
    #         "samples",
    #         columns=["epoch", "temperatue", "text"],
    #         data=[[self.epoch, self.model.temperature, generated_output]],
    #     )
    #     wandb.log({"train/text": generated_output})
    #
    #     return generated_output
