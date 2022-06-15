import math
from pathlib import Path
from typing import List, Optional

import flair
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from rich import print
from rich.panel import Panel
from torch import nn

import wandb
from src.data import tokenize_batch
from src.loss import CrossEntropyLossSoft
from src.models.ngme import NGramsEmbedding, soft_n_hot


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(pl.LightningModule):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(pl.LightningModule):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(
        self,
        dictionary,
        embedding_size,
        nhead,
        nhid,
        nlayers,
        ngrams,
        unk_t,
        is_forward_lm=True,
        document_delimiter="\n",
        dropout=0.5,
        gen_args: dict = {},
        unigram_ppl: bool = False,
        weighted_loss: bool = False,
        weighted_labels: bool = False,
        n_pos_embeddings: bool = False
    ):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError(
                "TransformerEncoder module does not exist in PyTorch 1.1 or lower."
            )

        self.ntoken = len(dictionary)
        self.dictionary = dictionary
        self.is_forward_lm = is_forward_lm
        self.ngrams = ngrams
        self.unk_t = unk_t
        self.nlayers = nlayers
        self.document_delimiter = document_delimiter
        self.dropout = dropout

        self.hidden_size = nhid
        self.nhead = nhead

        self.model_type = "Transformer"
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embedding_size, dropout)
        encoder_layers = TransformerEncoderLayer(
            embedding_size, nhead, self.hidden_size, dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = NGramsEmbedding(self.ntoken, embedding_size)
        self.embedding_size = embedding_size
        self.decoder = nn.Linear(embedding_size, self.ntoken)

        self.unigram_ppl = unigram_ppl
        self.weighted_labels = weighted_labels
        if weighted_loss:
            self.criterion = CrossEntropyLossSoft(weight=self.dictionary.create_weight_tensor())
            print(self.criterion.weight)
        else:
            self.criterion = CrossEntropyLossSoft()

        self.init_weights()
        self.epoch = 0

        for key, value in gen_args.items():
            setattr(self, key, value)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return optimizer  # ], [lr_scheduler]

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(src.size(1)).to(
                    self.device
                )
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.encoder(src) * math.sqrt(self.embedding_size)
        print(src.size())
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)

    def training_step(self, batch, batch_idx):
        ntokens = len(self.dictionary)
        output = self.forward(batch["source"])
        output = output.view(-1, ntokens)
        target = soft_n_hot(batch["target"], ntokens, self.weighted_labels)
        target = target.view(-1, ntokens)
        loss = self.criterion(output, target)
        self.log("train/loss", loss)
        self.log("train/ppl", math.exp(loss), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ntokens = len(self.dictionary)
        output = self.forward(batch["source"])
        output = output.view(-1, ntokens)
        target = soft_n_hot(batch["target"], ntokens, self.weighted_labels)
        target = target.view(-1, ntokens)
        loss = self.criterion(output, target)
        self.log("val/loss", loss)
        self.log("val/ppl", math.exp(loss))

    def test_step(self, batch, batch_idx):
        ntokens = len(self.dictionary)
        output = self.forward(batch["source"])
        output = output.view(-1, ntokens)
        target = soft_n_hot(batch["target"], ntokens, self.weighted_labels)
        target = target.view(-1, ntokens)
        loss = self.criterion(output, target)
        self.log("test/loss", loss)
        self.log("test/ppl", math.exp(loss))

    def training_epoch_end(self, outputs) -> None:

        self.epoch += 1

        if self.generate:
            print(Panel(self.generate_text(), title="[green]Generated text"))
            self.train()

    def generate_text(self) -> str:

        if self.ngrams > 2:
            unk_idxs = set(
                [
                    self.dictionary.word2idx[f"<{i}-UNK>"]
                    for i in range(3, self.ngrams + 1)
                ]
            )
            token_idxs = set(self.dictionary.word2idx.values())
            token_idxs = torch.tensor(
                list(token_idxs.difference(unk_idxs)), dtype=torch.int64
            )

        ntokens = len(self.dictionary)

        inp = torch.randint(
            ntokens, (self.ngrams, 1, 1), dtype=torch.long, device=self.device
        )
        generated_output = self.dictionary.idx2word[inp[0][0].item()]

        with torch.no_grad():
            self.eval()
            for i in range(self.chars):

                output = self(inp, False)

                # Only use the generated ngrams
                output = output[-1]

                if self.temperature == 0.0:
                    # output = F.softmax(output, dim=0).cpu()
                    # Just get highest confidence
                    ngram_idx = torch.argmax(output)
                else:
                    # output = F.log_softmax(output, dim=0)

                    # Remove all UNK tokens for ngram > 2
                    if self.ngrams > 2:
                        output = torch.index_select(output, 0, token_idxs).cpu()

                    word_weights = (
                        output[-1].squeeze().div(self.temperature).exp().cpu()
                    )
                    # multinomial over all tokens
                    ngram_idx = torch.multinomial(word_weights, 1)[0]

                # Get ngram word
                word = self.dictionary.idx2word[ngram_idx]

                # Append to generated sequence
                generated_output = generated_output + word

                # Use last 200 chars as sequence for new input
                inp = (
                    self.dictionary.tokenize_line(
                        generated_output[-200:],
                    )["source"]
                    .unsqueeze(dim=2)
                    .to(self.device)
                )
                # print(inp.size())

            self.train()

        self.logger.log_text(
            "samples",
            columns=["epoch", "temperatue", "text"],
            data=[[self.epoch, self.temperature, generated_output]],
        )
        wandb.log({"train/text": generated_output})

        return generated_output

    def __getstate__(self):

        # serialize the language models and the constructor arguments (but nothing else)
        model_state = {
            "state_dict": self.state_dict(),
            "dictionary": self.dictionary,
            "is_forward_lm": self.is_forward_lm,
            "hidden_size": self.hidden_size,
            "nlayers": self.nlayers,
            "embedding_size": self.embedding_size,
            "document_delimiter": self.document_delimiter,
            "dropout": self.dropout,
            "ngrams": self.ngrams,
            "unk_t": self.unk_t,
            "nhead": self.nhead,
        }

        return model_state

    def __setstate__(self, d):

        # special handling for deserializing language models
        if "state_dict" in d:

            # re-initialize language model with constructor arguments
            language_model = TransformerModel(
                dictionary=d["dictionary"],
                nlayers=d["nlayers"],
                ngrams=d["ngrams"],
                nhid=d["hidden_size"],
                unk_t=d["unk_t"],
                # nout=d["nout"],
                embedding_size=d["embedding_size"],
                is_forward_lm=d["is_forward_lm"],
                document_delimiter=d["document_delimiter"],
                dropout=d["dropout"],
                nhead=d["nhead"],
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

    def save(self, file):
        # TODO: Can we make this flair compatible?
        model_state = {
            "state_dict": self.state_dict(),
            "dictionary": self.dictionary,
            "is_forward_lm": self.is_forward_lm,
            "hidden_size": self.hidden_size,
            "nlayers": self.nlayers,
            "embedding_size": self.embedding_size,
            "document_delimiter": self.document_delimiter,
            "dropout": self.dropout,
            "ngrams": self.ngrams,
            "unk_t": self.unk_t,
            "nhead": self.nhead,
        }

        if isinstance(file, str):
            file = Path(file)

        # Prepend "flair_" prefix
        file = file.parent / ("flair_" + str(file.name))

        print(f"Save flair model state: {str(file)}")
        torch.save(model_state, str(file), pickle_protocol=4)

    def forward2(self, input, ordered_sequence_lengths=None):

        encoded = self.encoder(input) * math.sqrt(self.embedding_size)
        encoded = self.pos_encoder(encoded)

        output = self.transformer_encoder(encoded, self.src_mask)
        decoded = self.decoder(output)

        # decoded = self.decoder(
        #     output.view(output.size(0) * output.size(1), output.size(2))
        # )

        # output: [seq_len, batch_size, hidden]
        return decoded, output

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

        batches: List[torch.Tensor] = []

        # push each chunk through the RNN language model
        for chunk in chunks:
            len_longest_chunk: int = len(max(chunk, key=len))
            sequences_as_char_indices: List[torch.Tensor] = []

            for string in chunk:
                chars = list(string) + [" "] * (len_longest_chunk - len(string))

                chars = ["".join(chars)]

                # [ngram, 1, sequence]
                n_gram_char_indices = tokenize_batch(
                    self.dictionary, chars, self.ngrams, otf=True
                ).unsqueeze(dim=1)

                sequences_as_char_indices.append(n_gram_char_indices)

            # [ngram, batch_size, sequence]
            batches.append(torch.cat(sequences_as_char_indices, dim=1))

        output_parts = []
        for batch in batches:
            # [ngram, sequence, batch_size]
            batch = batch.transpose(1, 2).to(flair.device)

            _, transformer_output = self.forward2(batch)

            output_parts.append(transformer_output)

        # concatenate all chunks to make final output
        output = torch.cat(output_parts)

        return output
