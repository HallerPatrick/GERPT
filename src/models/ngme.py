from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

n_dists = {
    0: [1],
    1: [0.4, 0.6],
    2: [0.2, 0.3, 0.5],
    3: [0.1, 0.2, 0.3, 0.4],
    4: [0.1, 0.15, 0.2, 0.25, 0.3],
}


def n_hot(t, num_clases):
    shape = list(t.size())[1:]

    shape.append(num_clases)
    ret = torch.zeros(shape).to(t.device)

    # Expect that first dimension is for all n-grams
    for seq in t:
        ret.scatter_(-1, seq.unsqueeze(-1), 1)

    return ret


@lru_cache(maxsize=5)
def soft_dist(n):
    return [1 / n] * n


@lru_cache(maxsize=5)
def soft_weigthed_n_dist(n):
    return n_dists[n - 1]


def soft_n_hot(input, num_classes, weighted=False):
    # soft_dist = 1 / input.size(0)

    shape = list(input.size())[1:]

    shape.append(num_classes)

    ret = torch.zeros(shape).to(input.device)

    if weighted:
        soft_labels = soft_weigthed_n_dist(input.size()[0])
    else:
        soft_labels = soft_dist(input.size()[0])

    for i, t in enumerate(input):
        ret.scatter_(-1, t.unsqueeze(-1), soft_labels[i])

    return ret


class NGramsEmbedding(nn.Embedding):
    """N-Hot encoder"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight,
            device=device,
            dtype=dtype,
        )

        # self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.num_classes = num_embeddings

    def forward(self, input: torch.Tensor, **kwargs):
        return self._forward(n_hot(input, self.num_classes, **kwargs))

    def _forward(self, n_hot: torch.Tensor) -> torch.Tensor:
        return F.linear(n_hot, self.weight.t())
