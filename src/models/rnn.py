import math
import itertools
from pathlib import Path
from typing import List, Optional, Union

import flair
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero


import torch
import torch.nn as nn
import torch.nn.functional as F

from rich import print
from rich.panel import Panel
from src import utils

from src.loss import CrossEntropyLossSoft

# from src.utils import display_text

from .ngme import NGramsEmbedding, soft_n_hot


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class RNNModel(pl.LightningModule):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        dictionary,
        nlayers: int,
        ngrams: int,
        hidden_size: int,
        unk_t: int,
        nout=None,
        embedding_size: int = 100,
        is_forward_lm=True,
        document_delimiter: str = "\n",
        dropout=0.25,
        unigram_ppl: bool = False,
        weighted_loss: bool = False,
        weighted_labels: bool = False,
        strategy: str = "const",
        generate: bool = False,
        temperature: float = 0.7,
        chars_to_gen: int = 1000,
        cell_type = "lstm",
        packed = False
    ):
        super(RNNModel, self).__init__()

        self.ntokens = len(dictionary)

        self.encoder = NGramsEmbedding(len(dictionary), embedding_size, packed=packed)
        self.ngrams = ngrams
        self.unk_t = unk_t
        self.dictionary = dictionary
        self.nlayers = nlayers
        self.is_forward_lm = is_forward_lm
        self.nout = nout
        self.document_delimiter = document_delimiter
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.weighted_labels = weighted_labels
        self.weighted_loss = weighted_loss
        self.strategy = strategy
        self.unigram_ppl = unigram_ppl

        self.criterion = None
        self.bidirectional = False

        if cell_type == "lstm":
            if nlayers == 1 and not self.bidirectional:
                self.rnn = nn.LSTM(
                    embedding_size, hidden_size, nlayers, bidirectional=self.bidirectional
                )
            else:
                self.rnn = nn.LSTM(
                    embedding_size,
                    hidden_size,
                    nlayers,
                    dropout=dropout,
                    bidirectional=self.bidirectional,
                )
        elif cell_type == "mog_lstm":
            from src.models.mogrifier_lstm import MogLSTM
            # TODO: No Support for bidirectional and layers yet
            # TODO: Good amount of iterations for mog?
            self.rnn = MogLSTM(embedding_size, hidden_size, 5)


        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(
            hidden_size * 2 if self.bidirectional else hidden_size,
            self.ntokens
            if not self.unigram_ppl
            else len(self.dictionary.ngram2idx2word[1]),
        )

        self.save_hyperparameters()
        self.init_weights()

        self.hidden = None
        self.epoch = 0

        self.generate = generate
        self.temperature = temperature
        self.chars_to_gen = chars_to_gen

        self.proj = None

    def setup(self, stage) -> None:
        self.criterion = CrossEntropyLossSoft(
            weight=self.dictionary.create_weight_tensor(
                self.unigram_ppl, self.weighted_loss
            ),
        )

    @staticmethod
    def initialize(matrix):
        in_, out_ = matrix.size()
        stdv = math.sqrt(3.0 / (in_ + out_))
        matrix.detach().uniform_(-stdv, stdv)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.SGD(self.parameters(), lr=20.0)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.25,
            verbose=True,
            min_lr=1.25,
            threshold=1e-6,
            patience=10000,
            threshold_mode="abs",
        )
        return [optimizer], [
            {"scheduler": lr_scheduler, "monitor": "train/loss", "interval": "step"}
        ]

    def _step(self, batch):

        # batch_size = batch["source"].size()[-1]
        source, target = batch[0].permute((1, 2, 0)), batch[1].permute((1, 2, 0))

        batch_size = source.size(2)

        if not self.hidden:
            self.hidden = self.init_hidden(batch_size)

        decoded, hidden = self.forward(source, self.hidden)
        self.hidden = repackage_hidden(hidden)

        if self.unigram_ppl:
            target = soft_n_hot(
                target,
                len(self.dictionary.ngram2idx2word[1]),
                self.strategy,
                self.weighted_labels,
                self.unigram_ppl,
                self.encoder.packed
            )
            target = target.view(-1, len(self.dictionary.ngram2idx2word[1]))
        else:
            target = soft_n_hot(
                target,
                self.ntokens,
                self.strategy,
                self.weighted_labels,
                self.unigram_ppl,
                self.encoder.packed
            )
            target = target.view(-1, self.ntokens)
            
        loss = self.criterion(decoded, target)
        return loss, decoded, target

    def training_step(self, batch, batch_idx):
        loss, decoded, target = self._step(batch)

        self.log("train/loss", loss)
        try:
            self.log("train/ppl", math.exp(loss), prog_bar=True)
        except:
            pass

        # Unigram output
        output = torch.index_select(
            decoded,
            1,
            torch.tensor(list(self.dictionary.ngram2idx2word[1].keys())).to(
                self.device
            ),
        )
        targets = torch.index_select(
            target,
            1,
            torch.tensor(list(self.dictionary.ngram2idx2word[1].keys())).to(
                self.device
            ),
        )
        # Back to simple one hot encoding
        targets[targets != 0] = 1

        unigram_loss = self.criterion.unigram_loss(output, targets)

        self.log("train/unigram_loss", unigram_loss, prog_bar=True)
        self.log("train/unigram_ppl", math.exp(unigram_loss), prog_bar=True)

        return loss

    def validation_step(self, batch, _):
        loss, _, _ = self._step(batch)

        self.log("valid/loss", loss, sync_dist=True)
        self.log("valid/ppl", math.exp(loss), prog_bar=True, sync_dist=True)
        self.log("valid/bpc", loss / np.log(2), sync_dist=True)

    def test_step(self, batch, _):
        loss, _, _ = self._step(batch)

        self.log("test/loss", loss)
        self.log("test/ppl", math.exp(loss), prog_bar=True)

    def training_epoch_end(self, _) -> None:

        self.epoch += 1

        if self.generate:
            # Only rank zero gives output
            result= self.generate_text()
            if result:
                print(Panel(result, title="[green]Generated text"))
            self.train()
        # Reset hidden after each epoch
        self.hidden = None

    @rank_zero.rank_zero_only
    def generate_text(self) -> str:

        inp = torch.randint(
            self.ntokens, (self.ngrams, 1, 1), dtype=torch.int64, device=self.device
        )
        idx = inp[0][0].detach()
        generated_output = self.dictionary.get_item_for_index(idx.item())
        sample_text = self.dictionary.get_item_for_index(idx.item())


        with torch.no_grad():
            self.eval()
            for i in range(self.chars_to_gen):

                # Reset hidden
                hidden = self.init_hidden(1)
                output, hidden = self(inp, hidden)

                # Only use the generated ngrams
                output = output[-1]

                if self.temperature == 0.0:
                    output = F.softmax(output, dim=0).detach()
                    # Just get highest confidence
                    ngram_idx = torch.argmax(output)
                    word = self.dictionary.get_item_for_index(ngram_idx.item())
                else:
                    output = F.log_softmax(output, dim=0)

                    word_weights = output.squeeze().div(self.temperature).exp().detach()

                    # multinomial over all tokens
                    ngram_idx = torch.multinomial(word_weights, 1)[0]


                    ngram_order = self.dictionary.get_ngram_order(ngram_idx.item())
                    ngrams_idxs = [ngram_idx]
                    if self.dictionary.ngme == "sparse":
                        for i in range(1, ngram_order):
                            ngram_subset = torch.index_select(
                                word_weights,
                                0,
                                torch.tensor(list(self.dictionary.ngram2idx2word[i].keys()))
                            )

                            ngrams_idxs.append(torch.multinomial(ngram_subset, 1)[0])


                    word = "".join(list(reversed([self.dictionary.get_item_for_index(idx.item()) for idx in ngrams_idxs])))

                if word == "<pad>":
                    word = " "

                # Append to generated sequence
                generated_output = generated_output + word
                sample_text = sample_text + "·" + word

                # Use last 200 chars as sequence for new input
                
                inp = (
                    self.dictionary.tokenize_line(
                        list(generated_output[-200:]), id_type=torch.int64,
                        return_tensor="pt"
                    )["source"]
                    .unsqueeze(dim=2)
                    .to(self.device)
                )

            self.train()
        try:
            self.logger.log_text(
                "samples",
                columns=["epoch", "temperatue", "text"],
                data=[[self.epoch, self.temperature, sample_text]],
            )
        except:
            pass
        # wandb.log({"train/text": generated_output})

        return sample_text

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        # [#ngram, #seq_len, #batch_size]
        emb = self.encoder(input)
        emb = self.drop(emb)
        
        if isinstance(self.rnn, nn.LSTM):
            self.rnn.flatten_parameters()
            output, hidden = self.rnn(emb, hidden)
        else:
            emb = torch.transpose(emb, 0, 1).contiguous()
            print(emb.size())
            print(hidden[0].size())
            print(hidden[1].size())
            output, hidden = self.rnn(emb, hidden)

        decoded = self.decoder(output)

        if self.unigram_ppl:
            decoded = decoded.view(-1, len(self.dictionary.ngram2idx2word[1]))
        else:
            decoded = decoded.view(-1, self.ntokens)

        return decoded, hidden

    def forward2(self, input, hidden, ordered_sequence_lengths=None):

        # input: [ngram, sequence_length, batch]
        encoded = self.encoder(input)
        encoded = self.drop(encoded)

        self.rnn.flatten_parameters()

        output, hidden = self.rnn(encoded, hidden)

        if self.proj is not None:
            output = self.proj(output)

        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )

        # output: [seq_len, batch_size, ntokens]
        return (
            decoded.view(output.size(0), output.size(1), decoded.size(1)),
            output,
            hidden,
        )

    def init_hidden(self, bsz):
        weight = next(self.parameters()).detach()
        return (
            weight.new(
                self.nlayers * 2 if self.bidirectional else self.nlayers,
                bsz,
                self.hidden_size,
            )
            .zero_()
            .clone()
            .detach(),
            weight.new(
                self.nlayers * 2 if self.bidirectional else self.nlayers,
                bsz,
                self.hidden_size,
            )
            .zero_()
            .clone()
            .detach(),
        )

    def __getstate__(self):

        # serialize the language models and the constructor arguments (but nothing else)
        model_state = {
            "state_dict": self.state_dict(),
            "dictionary": self.dictionary,
            "is_forward_lm": self.is_forward_lm,
            "hidden_size": self.hidden_size,
            "nlayers": self.nlayers,
            "embedding_size": self.embedding_size,
            "nout": self.nout,
            "document_delimiter": self.document_delimiter,
            "dropout": self.dropout,
            "ngrams": self.ngrams,
            "unk_t": self.unk_t,
        }

        print(model_state)

        return model_state

    def __setstate__(self, d):

        # special handling for deserializing language models
        if "state_dict" in d:

            # re-initialize language model with constructor arguments
            language_model = RNNModel(
                dictionary=d["dictionary"],
                nlayers=d["nlayers"],
                ngrams=d["ngrams"],
                hidden_size=d["hidden_size"],
                unk_t=d["unk_t"],
                nout=d["nout"],
                embedding_size=d["embedding_size"],
                is_forward_lm=d["is_forward_lm"],
                document_delimiter=d["document_delimiter"],
                dropout=d["dropout"],
            )

            language_model.load_state_dict(d["state_dict"])

            # copy over state dictionary to self
            for key in language_model.__dict__.keys():
                self.__dict__[key] = language_model.__dict__[key]

            # set the language model to eval() by default (this is necessary since FlairEmbeddings "protect" the LM
            # in their "self.train()" method)
            self.eval()

        else:
            self.__dict__ = d

    def save(self, file: Union[str, Path]):

        model_state = {
            "state_dict": self.state_dict(),
            "dictionary": self.dictionary,
            "is_forward_lm": self.is_forward_lm,
            "hidden_size": self.hidden_size,
            "nlayers": self.nlayers,
            "embedding_size": self.embedding_size,
            "nout": self.nout,
            "document_delimiter": self.document_delimiter,
            "dropout": self.dropout,
            "ngrams": self.ngrams,
            "unk_t": self.unk_t,
        }

        if isinstance(file, str):
            file = Path(file)

        # Prepend "flair_" prefix
        file = file.parent / ("flair_" + str(file.name))

        print(f"Save flair model state: {str(file)}")
        torch.save(model_state, str(file), pickle_protocol=4)

    def get_representation(
        self,
        strings: List[str],
        start_marker: str,
        end_marker: str,
        chars_per_chunk: int = 512,
    ):

        len_longest_str: int = len(max(strings, key=len))

        # pad strings with whitespaces to longest sentence
        padded_strings: List[str] = []

        for string in strings:
            if not self.is_forward_lm:
                string = string[::-1]

            padded = f"{start_marker}{string}{end_marker}"
            padded_strings.append(padded)

        # cut up the input into chunks of max charlength = chunk_size
        chunks = []
        splice_begin = 0
        longest_padded_str: int = len_longest_str + len(start_marker) + len(end_marker)
        for splice_end in range(chars_per_chunk, longest_padded_str, chars_per_chunk):
            chunks.append([text[splice_begin:splice_end] for text in padded_strings])
            splice_begin = splice_end

        chunks.append(
            [text[splice_begin:longest_padded_str] for text in padded_strings]
        )

        hidden = self.init_hidden(len(chunks[0]))

        batches: List[torch.Tensor] = []

        # push each chunk through the RNN language model
        for chunk in chunks:
            len_longest_chunk: int = len(max(chunk, key=len))
            sequences_as_char_indices: List[torch.Tensor] = []

            for string in chunk:
                chars = list(string) + [" "] * (len_longest_chunk - len(string))

                chars = "".join(chars)

                # [ngram, 1, sequence]
                # self.dictionary.ngme = "dense"
                n_gram_char_indices = self.dictionary.tokenize_line(
                    chars, id_type=torch.int64
                )["source"].unsqueeze(dim=1)

                sequences_as_char_indices.append(n_gram_char_indices)

            # [ngram, batch_size, sequence]
            batches.append(torch.cat(sequences_as_char_indices, dim=1))

        output_parts = []
        for batch in batches:
            # batch: [ngram, batch, seq]
            batch = batch.transpose(1, 2).to(flair.device)

            # batch: [ngram, sequence, batch_size]
            _, rnn_output, hidden = self.forward2(batch, hidden)

            # rnn_output: [seq_len, batch, hidden]

            output_parts.append(rnn_output)

        # concatenate all chunks to make final output
        output = torch.cat(output_parts)

        return output
