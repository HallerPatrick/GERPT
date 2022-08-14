from typing import Optional


import torch

from transformers import PretrainedConfig


class TransformerConfig(PretrainedConfig):

    model_type = "ngme-transformer"

    def __init__(
        self,
        # dictionary=None,
        ntoken=1,
        embedding_size=256,
        nhead=2,
        nhid=256,
        nlayers=1,
        ngrams=1,
        unk_t=30,
        dropout=0.5,
        unigram_ppl: bool = False,
        weighted_loss: bool = False,
        weight_tensor: Optional[torch.Tensor] = None,
        weighted_labels: bool = False,
        **kwargs,
    ):
        
        self.ntoken = ntoken
        self.embedding_size = embedding_size
        self.nhead = nhead
        self.hidden_size = nhid
        self.nlayers = nlayers
        self.ngrams = ngrams
        self.unk_t = unk_t
        self.dropout = dropout
        self.unigram_ppl = unigram_ppl
        self.weighted_loss = weighted_loss
        self.weight_tensor = weight_tensor
        self.weighted_labels = weighted_labels

        super().__init__(**kwargs)

    @classmethod
    def from_args(cls, args,):
        return cls(
            ntoken=args.ntoken,
            embedding_size=args.embedding_size,
            nhead=args.nhead,
            nhid=args.hidden_size,
            nlayers=args.nlayers,
            ngrams=args.ngram,
            unk_t=args.unk_threshold,
            unigram_ppl=args.unigram_ppl,
            weighted_loss=args.weighted_loss,
            weighted_labels=args.weighted_labels,
        )
