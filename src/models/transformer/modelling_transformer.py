import math
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import PreTrainedModel
import flair

from src.loss import CrossEntropyLossSoft
from src.models.ngme import NGramsEmbedding
from src.models.transformer.configuration_transformer import TransformerConfig


class TransformerTransformer(PreTrainedModel):

    config_class = TransformerConfig

    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.ntoken = config.ntoken
        self.ngrams = config.ngrams
        self.unk_t = config.unk_t
        self.nlayers = config.nlayers
        self.dropout = config.dropout

        self.hidden_size = config.hidden_size
        self.nhead = config.nhead

        self.model_type = "Transformer"
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(config.embedding_size, config.dropout)
        encoder_layers = TransformerEncoderLayer(
            config.embedding_size, config.nhead, self.hidden_size, config.dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.nlayers)
        self.encoder = NGramsEmbedding(self.ntoken, config.embedding_size)
        self.embedding_size = config.embedding_size
        self.decoder = nn.Linear(config.embedding_size, self.ntoken)

        self.unigram_ppl = config.unigram_ppl

        self.weighted_labels = config.weighted_labels

        self.criterion = CrossEntropyLossSoft(
            weight=torch.tensor(config.weight_tensor).to(self.device)
        )

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True, **kwargs):

        # (ngram, seq, batch)
        if has_mask:
            if self.src_mask is None or self.src_mask.size(0) != src.size(1):
                mask = self._generate_square_subsequent_mask(src.size(1)).to(
                    self.device
                )
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.embedding_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def forward_hidden(self, src, has_mask=True, **kwargs):
        # if "attention_mask" in kwargs:
        #     src = src.squeeze(0).unsqueeze(-1)

        if has_mask:
            if self.src_mask is None or self.src_mask.size(0) != src.size(1):
                mask = self._generate_square_subsequent_mask(src.size(1)).to(
                    self.device
                )
                self.src_mask = mask
        else:
            self.src_mask = None
        
        # in: [ngram, seq, batch]

        src = self.encoder(src)
        # out: [seq, batch, emb]

        src = src * math.sqrt(self.embedding_size)
        # out: [seq, batch, emb]

        src = self.pos_encoder(src)
        # out: [seq, batch, emb]

        output = self.transformer_encoder(src, self.src_mask)

        return output

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

                chars = "".join(chars)

                # [ngram, 1, sequence]
                # self.dictionary.ngme = "dense"
                n_gram_char_indices = self.tokenizer(
                    chars, return_attention_mask=False, return_tensors="pt"
                )["input_ids"]

                sequences_as_char_indices.append(n_gram_char_indices)

            # [ngram, batch_size, sequence]
            batches.append(torch.cat(sequences_as_char_indices, dim=0))

        output_parts = []
        for batch in batches:
            
            # batch: [batch, ngram, seq]
            batch = torch.permute(batch, ( 1, 2, 0 )).to(flair.device)
            # batch: [batch, sequence, batch_size]
        
            # hidden: [seq_len, ngram, hidden]
            hidden = self.forward_hidden(batch)
            
            output_parts.append(hidden)

        # concatenate all chunks to make final output
        output = torch.cat(output_parts)

        return output


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

