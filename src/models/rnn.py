from collections.abc import Callable
import math
from pathlib import Path
from typing import List, Union

import flair
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero

from src.loss import AdaptiveLogSoftmaxWithLossSoft, CrossEntropyLossSoft

from .ngme import NGramsEmbedding, soft_n_hot, CanineEmbeddings


DEBUG = False


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def bce_loss(pred, target):
    pred = F.logsigmoid(pred)
    loss = torch.mean(-(target * torch.log(pred) + (1 - target) * torch.log(1 - pred)))
    return loss


class Decoder(pl.LightningModule):
    
    def __init__(self, weight) -> None:
        super().__init__()
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(self.weight.size(0)))
    
    def forward(self, x):
        return torch.mm(x, self.weight.t()) + self.bias


class RNNModel(pl.LightningModule):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        dictionary,
        nlayers: int,
        ngrams: int,
        hidden_size: int,
        nout=None,
        embedding_size: int = 100,
        lr: float = 20,
        is_forward_lm=True,
        document_delimiter: str = "\n",
        dropout=0.25,
        weighted_loss: bool = False,
        weighted_labels: bool = False,
        strategy: str = "const",
        generate: bool = False,
        temperature: float = 0.7,
        chars_to_gen: int = 1000,
        cell_type: str = "lstm",
        packed: bool = False,
        loss_type: str = "cross_entropy",
        has_decoder: bool = True
    ):
        super(RNNModel, self).__init__()

        self.ntokens = len(dictionary)
        # self.encoder = NGramsEmbedding(
        #     len(dictionary), embedding_size, packed=packed, freeze=True
        # )
        self.has_decoder = has_decoder
        self.encoder = CanineEmbeddings(embedding_size, 4, 5_000)
        print(self.encoder)

        self.ngrams = ngrams
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
        self.lr = lr
        self.loss_type = loss_type

        self.criterion = None
        self.bidirectional = False

        if cell_type == "lstm":
            if nlayers == 1 and not self.bidirectional:
                self.rnn = nn.LSTM(
                    embedding_size,
                    hidden_size,
                    nlayers,
                    bidirectional=self.bidirectional,
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
        
        # self.decoder = Decoder(self.encoder.weight)
        if loss_type != "adaptive_softmax":
            self.decoder = nn.Linear(
                hidden_size * 2 if self.bidirectional else hidden_size, self.ntokens
            )
        else:
            self.has_decoder = False
            self.decoder = None
        #
        # self.decoder.weight = self.encoder.weight

        self.save_hyperparameters()
        self.init_weights()

        self.hidden = None
        self.epoch = 0

        self.generate = generate
        self.temperature = temperature
        self.chars_to_gen = chars_to_gen

        self.proj = None

    def setup(self, stage) -> None:

        if self.loss_type == "cross_entropy":
            self.criterion = CrossEntropyLossSoft(
                weight=self.dictionary.create_weight_tensor(self.weighted_loss),
            )
        elif self.loss_type == "adaptive_softmax":

            # Using len of ngram vocabs as cutoffs
            vocabs = self.dictionary.vocab_size_per_ngram()

            del vocabs[-1]
            cutoffs = [vocabs[0]]
            for i, vocab in enumerate(vocabs[1:]):
                cutoffs.append(vocab + cutoffs[i])

            self.criterion = AdaptiveLogSoftmaxWithLossSoft(
                in_features=self.hidden_size,
                n_classes=self.ntokens,
                cutoffs=cutoffs,
                weight=self.dictionary.create_weight_tensor(self.weighted_loss)
            )

        elif self.loss_type == "split_cross_entropy":
            # TODO: Ideally we apply the same weights as for normal cross_entropy to all subsets
            self.criterion = []
            weights = self.dictionary.create_weight_tensor(self.weighted_loss)
            weights = self._subset_logits(weights.unsqueeze(0))
            for weight_tensor in weights:
                self.criterion.append(
                    nn.CrossEntropyLoss(weight=weight_tensor.squeeze(0))
                )
        else:
            raise ValueError("Loss not supported")

    @staticmethod
    def initialize(matrix):
        in_, out_ = matrix.size()
        stdv = math.sqrt(3.0 / (in_ + out_))
        matrix.detach().uniform_(-stdv, stdv)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[4, 5, 6, 7], gamma=0.25, verbose=True
        # )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.25, verbose=True, patience=120
        )
        return [optimizer], [{"scheduler": lr_scheduler, "monitor": "train/loss"}]

    def _debug_input(self, source, target):
        print("Source:")
        self.dictionary.print_batch(source)
        print("Target:")
        self.dictionary.print_batch(target)
        input("continue")

    def _step(self, batch):
        source, target = batch[0].permute((1, 2, 0)), batch[1].permute((1, 2, 0))

        if DEBUG:
            self._debug_input(source, target)

        if not self.hidden:
            batch_size = source.size(2)
            self.hidden = self.init_hidden(batch_size)

        decoded, hidden = self.forward(source, self.hidden)
        self.hidden = repackage_hidden(hidden)

        # target: [ngram, seq, batch]
        if self.loss_type == "cross_entropy":
            target = soft_n_hot(
                target,
                self.ntokens,
                # self.strategy,
                "linear",
                self.weighted_labels,
                False,
                # self.encoder.packed,
            )
            # [seq, vocab]
            target = target.reshape((-1, target.size(-1)))
            loss = self.criterion(decoded, target)
            # 158 secs
        elif self.loss_type == "split_cross_entropy":
            target = target.reshape((target.size(0), -1))
            loss = self.subset_cross_entropy(decoded, target)

        elif self.loss_type == "adaptive_softmax":
            target = target.reshape((target.size(0), -1))
            decoded = decoded.reshape((-1, decoded.size(-1)))
            output, loss = self.criterion(decoded, target)
        else:
            raise ValueError("Loss not supported")

        return loss, decoded, target

    def adaptive_softmax_loss(self, decoded, target):
        subsets = self._subset_logits(decoded)

        losses = []
        for i, (subset, n_gram_target) in enumerate(
            zip(subsets, self._map_n_gram_id(target))
        ):

            if self.criterion[i].head.weight.device != subset.device:
                self.criterion[i].head.to(subset.device)
                self.criterion[i].tail.to(subset.device)

            output, loss = self.criterion[i](subset, n_gram_target.reshape(-1))
            losses.append(loss)

        for n, loss in enumerate(losses):
            self.log(f"train/{n+1}-ngram-loss", loss)

        loss = sum(losses)
        return loss

    def subset_cross_entropy(self, decoded, target):
        subsets = self._subset_logits(decoded)

        losses = []
        for i, (subset, n_gram_target) in enumerate(
            zip(subsets, self._map_n_gram_id(target))
        ):

            if self.criterion[i].weight.device != subset.device:
                self.criterion[i].weight = self.criterion[i].weight.to(subset.device)
            
            loss = self.criterion[i](subset, n_gram_target) * (i+1)
            losses.append(loss)

        for n, loss in enumerate(losses):
            self.log(f"train/{n+1}-ngram-loss", loss)

        loss = sum(losses)
        return loss

    def _map_n_gram_id(self, tensor):
        """
        To calculate the cross entropy of our subset probabilities against target,
        we have to map the n-gram ids of the target tensor to the subset.

        E.g.:

            1-ngram: [0, 100],
            2-ngram: [101, 300],
            3-ngram: [301, 500]

            2-gram id: 101 -> 0
            2-gram id: 224 -> 123
        """

        subsets = []
        for n, tensor in enumerate(tensor):
            smallest_ngram_idx = list(self.dictionary.ngram2idx2word[n + 1].keys())[0]
            subset_tensor = torch.sub(tensor, smallest_ngram_idx)
            subsets.append(subset_tensor)

        return subsets

    def _subset_logits(self, logits) -> List[torch.Tensor]:
        """
        logits: [seq, vocab]
        """

        subsets = []

        for n_gram in range(self.dictionary.ngram):
            n_gram_idxs = torch.tensor(
                list(self.dictionary.ngram2idx2word[n_gram + 1].keys())
            ).to(logits.device)
            subset = torch.index_select(logits, 1, n_gram_idxs)
            subsets.append(subset)

        return subsets

    def training_step(self, batch, batch_idx):
        loss, decoded, target = self._step(batch)

        self.log("train/loss", loss)
        try:
            self.log("train/ppl", math.exp(loss), prog_bar=True)
        except:
            pass

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

        # --- Calculate Bits-per-Byte ---
        # 1. Reconstruct original text
        # 2. Count number of UTF-8 bytes
        # 3. Counter number of tokens (TODO: What tokens to pick?!)
        # 4. Forumlar: BPB = (L_T / T_B) * log_2(e**l)
        #    where: L_T, number of tokens, L_B number of UTF-8 encoded bytes, l = NLL
        # source = batch[0]
        # num_tokens = source.size(1)
        # num_bycross_entropytes = None
        # print(batch.size())

    def training_epoch_end(self, _) -> None:
        # Reset hidden after each epoch
        self.hidden = None

    # @rank_zero.rank_zero_only
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

                if self.temperature == 0.0:
                    output = output[-1]
                    output = F.softmax(output, dim=0).detach()
                    # Just get highest confidence
                    ngram_idx = torch.argmax(output)
                    word = self.dictionary.get_item_for_index(ngram_idx.item())
                else:
                    output = output[-1]

                    if self.loss_type in ["split_cross_entropy"]:
                        subsets = self._subset_logits(output.unsqueeze(0))

                        ngram_idxs = []
                        for n, subset in enumerate(subsets):
                            subset_output = F.log_softmax(subset.squeeze(0), dim=0)

                            word_weights = (
                                subset_output.squeeze()
                                .div(self.temperature)
                                .exp()
                                .detach()
                            )
                            subset_ngram_idx = torch.multinomial(word_weights, 1)[
                                0
                            ].item()
                            smallest_ngram_idx = list(
                                self.dictionary.ngram2idx2word[n + 1].keys()
                            )[0]
                            ngram_idx = subset_ngram_idx + smallest_ngram_idx
                            output_probability = word_weights[subset_ngram_idx].item()
                            ngram_idxs.append((ngram_idx, output_probability))

                        # ngram_idx = list(sorted(ngram_idxs, key=lambda x: x[1], reverse=True))[0][0]
                        ngram_idx = ngram_idxs[0][0]
                    elif self.loss_type == "adaptive_softmax":
                        output = self.criterion.log_prob(output)
                        output = F.log_softmax(output.squeeze(0), dim=0)
                        word_weights = (
                            output.squeeze().div(self.temperature).exp().detach()
                        )
                        ngram_idx = torch.multinomial(word_weights, 1)[0].item()
                    else:
                        # output = output[-1]
                        output = F.log_softmax(output, dim=0)

                        word_weights = (
                            output.squeeze().div(self.temperature).exp().detach()
                        )

                        # multinomial over all tokens
                        # To avoid unk gens, take ( n*unique_unks_in_dict )+1
                        # Probe for first non unk token
                        ngram_idx = torch.multinomial(word_weights, 1)[0].item()

                    ngram_order = self.dictionary.get_ngram_order(ngram_idx)
                    ngrams_idxs = [ngram_idx]
                    if self.dictionary.ngme == "sparse":
                        for i in range(1, ngram_order):
                            ngram_subset = torch.index_select(
                                word_weights,
                                0,
                                torch.tensor(
                                    list(self.dictionary.ngram2idx2word[i].keys())
                                ),
                            )

                            ngrams_idxs.append(torch.multinomial(ngram_subset, 1)[0])

                    word = "".join(
                        list(
                            reversed(
                                [
                                    self.dictionary.get_item_for_index(idx)
                                    for idx in ngrams_idxs
                                ]
                            )
                        )
                    )

                if word == "<pad>":
                    word = " "

                # Append to generated sequence
                generated_output = generated_output + word
                sample_text = sample_text + "·" + word

                # Use last 200 chars as sequence for new input

                inp = (
                    self.dictionary.tokenize_line(
                        list(generated_output[-200:]),
                        id_type=torch.int64,
                        return_tensor="pt",
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

        return sample_text

    def init_weights(self):
        initrange = 0.1

        if hasattr(self.encoder, "weight"):
            self.encoder.weight.detach().uniform_(-initrange, initrange)

        if self.has_decoder:
            self.decoder.bias.detach().fill_(0)
            self.decoder.weight.detach().uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        # [#ngram, #seq_len, #batch_size]
        emb = self.encoder(input)
        emb = self.drop(emb)

        self.rnn.flatten_parameters()
        output, hidden = self.rnn(emb, hidden)

        if self.has_decoder:
            output = self.decoder(output.view(-1, output.size(-1))) # .view(-1, self.ntokens)

        return output, hidden

    def forward2(self, input, hidden, ordered_sequence_lengths=None):
        # input: [ngram, sequence_length, batch]
        encoded = self.encoder(input)
        encoded = self.drop(encoded)

        self.rnn.flatten_parameters()

        output, hidden = self.rnn(encoded, hidden)

        if self.proj is not None:
            output = self.proj(output)
    
        if self.has_decoder:
            decoded = self.decoder(
                output.view(output.size(0) * output.size(1), output.size(2))
            )
            decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))
        else:
            decoded = None

        # output: [seq_len, batch_size, ntokens]
        return (
            decoded,
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
            "has_decoder": self.decoder is not None,
        }

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
                nout=d["nout"],
                embedding_size=d["embedding_size"],
                is_forward_lm=d["is_forward_lm"],
                document_delimiter=d["document_delimiter"],
                dropout=d["dropout"],
                has_decoder=d["has_decoder"]
            )
            
            print("Loading state dict")
            print(d)
            language_model.load_state_dict(d["state_dict"], strict=False)

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
            "has_decoder": self.has_decoder
        }

        if isinstance(file, str):
            file = Path(file)

        # Prepend "flair_" prefix
        file = file.parent / ("flair_" + str(file.name))

        print(f"Save flair model state: {str(file)}")
        torch.save(model_state, str(file), pickle_protocol=4)
        return str(file)

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

                # chars = "".join(chars)

                # [ngram, 1, sequence]
                # self.dictionary.ngme = "dense"
                n_gram_char_indices = self.dictionary.tokenize_line(
                    chars, id_type=torch.int64, return_tensor="pt"
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


try:
    import lm_eval
    from lm_eval.base import BaseLM

    from einops import rearrange

    def decode(dictionary, tokens):
        """Just take the unigram tokens and decode"""
        unigram_tokens = tokens[0]
        return "".join([dictionary.get_item_for_index(idx) for idx in unigram_tokens])

    class EvalRNNModel(BaseLM):
        def __init__(self, ckpt_path: str):
            super().__init__()
            self.model = RNNModel.load_from_checkpoint(ckpt_path)

        def _model_call(self, inps):
            # inps: [batch, ngram, sequence] -> [ngram, sequence, batch]
            inps = rearrange(inps, "b n s -> n s b")
            with torch.no_grad():
                output = self.model(inps)
                return output

        def tok_encode(self, string: str):
            return self.model.dictionary.tokenize_line(string)

        def tok_decode(self, tokens):
            return decode(self.model.dictionary, tokens)

        def _model_generate(self, context, max_length, eos_token_id):
            self.model.chars_to_gen = max_length
            return self.model.generate_text()

except ImportError:
    pass
