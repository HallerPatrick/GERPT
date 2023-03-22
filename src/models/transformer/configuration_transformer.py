from typing import Optional

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
        dropout=0.5,
        unigram_ppl: bool = False,
        weighted_loss: bool = False,
        weight_tensor: Optional[list] = None,
        ngram_indexes: Optional[dict] = None,
        weighted_labels: bool = False,
        generate: bool = False,
        temperature: float = 1.0,
        chars: int = 1000,
        pad_token_id: int = -1,
        **kwargs,
    ):
        self.ntoken = ntoken
        self.embedding_size = embedding_size
        self.nhead = nhead
        self.hidden_size = nhid
        self.nlayers = nlayers
        self.ngrams = ngrams
        self.dropout = dropout
        self.unigram_ppl = unigram_ppl
        self.weighted_loss = weighted_loss
        self.weight_tensor = weight_tensor
        self.weighted_labels = weighted_labels
        self.ngram_indexes = ngram_indexes
        self.generate = generate
        self.temperature = temperature
        self.chars = chars
        self.pad_token_id = pad_token_id

        super().__init__(**kwargs)

    @classmethod
    def from_args(cls, args):
        return cls(
            ntoken=args.ntoken,
            embedding_size=args.embedding_size,
            nhead=args.nhead,
            nhid=args.hidden_size,
            nlayers=args.nlayers,
            ngrams=args.ngram,
            unigram_ppl=args.unigram_ppl,
            weighted_loss=args.weighted_loss,
            weight_tensor=args.weight_tensor,
            weighted_labels=args.weighted_labels,
            ngram_indexes=args.ngram_indexes,
            generate=args.generate,
            temperature=args.temperature,
            chars=args.chars,
            # pad_token_id=args.pad_token_id,
        )
