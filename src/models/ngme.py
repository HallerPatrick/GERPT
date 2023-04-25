from functools import lru_cache
from typing import Optional, List
from math import log

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src import utils

try:
    from ngme_cpp import n_hot
except ImportError:

    def n_hot(t, num_clases, packed=False):
        if packed:
            t = utils.unpack_batched_tensor(t)

        shape = list(t.size())[1:]

        shape.append(num_clases)
        ret = torch.zeros(shape).to(t.device)

        # Expect that first dimension is for all n-grams
        for seq in t:
            ret.scatter_(-1, seq.unsqueeze(-1), 1)

        return ret


n_dists = {
    0: [1],
    1: [0.4, 0.6],
    2: [0.2, 0.3, 0.5],
    3: [0.1, 0.2, 0.3, 0.4],
    4: [0.1, 0.15, 0.2, 0.25, 0.3],
}

strats = {"linear": lambda x: x, "log": lambda x: log(x + 1), "exp": lambda x: x ** 2}


@lru_cache(maxsize=5)
def soft_dist(n):
    return [1 / n] * n


@lru_cache(maxsize=5)
def n_dist(n: int, strategy: str) -> list[float]:
    """dist of ngram weight is logarithmic"""
    ns = list(range(1, n + 1))
    xs = list(map(strats[strategy], ns))
    result = list(map(lambda x: x / sum(xs), xs))
    return result


def soft_n_hot(
    input,
    num_classes: int,
    strategy: str,
    weighted=False,
    packed=False,
):

    if packed:
        input = utils.unpack_batched_tensor(input)

    shape = list(input.size())[1:]

    shape.append(num_classes)

    ret = torch.zeros(shape).to(input.device)

    if weighted:
        soft_labels = n_dist(input.size(0), strategy)
    else:
        soft_labels = [1] * input.size(0)

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
        packed=False,
        freeze: bool = False,
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

        self.num_classes = num_embeddings
        self.packed = packed

        self.weight.requires_grad = not freeze

    def forward(self, input: torch.Tensor, **kwargs):
        return self._forward(
            n_hot(input, self.num_classes, **kwargs, packed=self.packed)
        )

    def _forward(self, n_hot: torch.Tensor) -> torch.Tensor:
        return F.linear(n_hot, self.weight.t())


# Support up to 16 hash functions.
_PRIMES = [31, 43, 59, 61, 73, 97, 103, 113, 137, 149, 157, 173, 181, 193, 211, 223]


class CanineEmbeddings(nn.Module):
    """Construct the character, position and token_type embeddings."""

    def __init__(
        self,
        embedding_size: int,
        num_hash_functions: int,
        num_hash_buckets: int,
        layer_norm_eps: float = 1e-6,
        hidden_dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = embedding_size
        self.num_hash_functions = num_hash_functions
        self.num_hash_buckets = num_hash_buckets
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob

        # character embeddings
        shard_embedding_size = self.hidden_size // self.num_hash_functions
        for i in range(self.num_hash_functions):
            name = f"HashBucketCodepointEmbedder_{i}"
            setattr(
                self, name, NGramsEmbedding(self.num_hash_buckets, shard_embedding_size)
            )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        # self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def _hash_bucket_tensors(self, input_ids, num_hashes: int, num_buckets: int):
        """
        Converts ids to hash bucket ids via multiple hashing.

        Args:
            input_ids: The codepoints or other IDs to be hashed.
            num_hashes: The number of hash functions to use.
            num_buckets: The number of hash buckets (i.e. embeddings in each table).

        Returns:
            A list of tensors, each of which is the hash bucket IDs from one hash function.
        """
        if num_hashes > len(_PRIMES):
            raise ValueError(f"`num_hashes` must be <= {len(_PRIMES)}")

        primes = _PRIMES[:num_hashes]

        result_tensors = []
        for prime in primes:
            hashed = ((input_ids + 1) * prime) % num_buckets
            result_tensors.append(hashed)
        return result_tensors

    def _embed_hash_buckets(
        self, input_ids, embedding_size: int, num_hashes: int, num_buckets: int
    ):
        """Converts IDs (e.g. codepoints) into embeddings via multiple hashing."""
        if embedding_size % num_hashes != 0:
            raise ValueError(
                f"Expected `embedding_size` ({embedding_size}) % `num_hashes` ({num_hashes}) == 0"
            )

        hash_bucket_tensors = self._hash_bucket_tensors(
            input_ids, num_hashes=num_hashes, num_buckets=num_buckets
        )
        embedding_shards = []
        for i, hash_bucket_ids in enumerate(hash_bucket_tensors):
            name = f"HashBucketCodepointEmbedder_{i}"
            shard_embeddings = getattr(self, name)(hash_bucket_ids)
            embedding_shards.append(shard_embeddings)

        return torch.cat(embedding_shards, dim=-1)

    def forward(
        self, input_ids: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:

        inputs_embeds = self._embed_hash_buckets(
            input_ids, self.hidden_size, self.num_hash_functions, self.num_hash_buckets
        )

        embeddings = self.LayerNorm(inputs_embeds)
        # embeddings = self.dropout(embeddings)
        return embeddings
