# Copyright (c) 2023 Authors of "Revisiting Deep Learning Models for Tabular Data"

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""The official package & illustrations for the NeurIPS 2021 paper "Revisiting Deep Learning Models for Tabular Data".

- `pip install paper_tabular_dl_revisiting_models`
- Paper: [arXiv](https://arxiv.org/abs/2106.11959)
- Code: [GitHub](https://github.com/yandex-research/tabular-dl-revisiting-models/blob/main/package)
- Example (with a Colab link inside):
  [GitHub](https://github.com/yandex-research/tabular-dl-revisiting-models/blob/main/package/example.ipynb)

# <span style="color:brown">What to expect from this package</span>

**Please, read this section.**

This package provides a minimal reimplementation and illustrations of
the main things used in the paper. In particular:
- This is NOT the code used to obtain the results in the paper.
  To reproduce the paper, see the instructions in *the root of the repository*.
- The package code aims to follow the paper code.
  *All differences with the paper code are explained in the source code of this package
  in the comments starting with `# NOTE: DIFF`.*
- Feel free to copy any part of the package source code and adjust it for your needs
  (please, keep the license header and/or add a link to this package).
- Adding new features is rather out of scope for this package.
  You can submit a feature request if you think that the change will be small.

# How to tune hyperparameters

- In the paper, for hyperparameter tuning, we used the
  [TPE sampler from Optuna](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html).
- The hyperparamer tuning spaces can be found in the appendix of the paper
  and in the `output/` directory in the main repository.
- For `FTTransformer`, there is also a default configuration for a quick start.

# API

.. note::
    Use the "View Source" buttons on the right to see the
    implementation of the module and individual items.

"""  # noqa: E501
__version__ = '0.0.4'

__all__ = [
    'MLP',
    'ResNet',
    'LinearEmbeddings',
    'CategoricalFeatureEmbeddings',
    'CLSEmbedding',
    'MultiheadAttention',
    'FTTransformerBackbone',
    'FTTransformer',
]


import math
import typing
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from torch.nn.parameter import Parameter

_INTERNAL_ERROR_MESSAGE = 'Internal error'


def _init_uniform_rsqrt(x: Tensor, d: int) -> None:
    # This is the initialization used in `torch.nn.Linear`.
    d_rsqrt = d**-0.5
    nn.init.uniform_(x, -d_rsqrt, d_rsqrt)


# NOTE: DIFF
# In the paper, for MLP, we tune the width of the first and the last layers separately
# from the rest of the layers. It turns out that using the same width for all layers
# is not worse, at least not on the datasets we use in the paper.
class MLP(nn.Module):
    """The MLP model from Section 3.1 in the paper.

    ```
    MLP:   (in) -> Block  -> ...  -> Block   -> (out)
    Block: (in) -> Linear -> ReLU -> Dropout -> (out)
    ```

    **Shape**

    - Input: `(*, d_in)`
    - Output: `(*, d_out or d_block)`

    **Examples**

    >>> batch_size = 2
    >>> x = torch.randn(batch_size, 3)
    >>> d_out = 1
    >>> m = MLP(
    ...    d_in=x.shape[1],
    ...    d_out=d_out,
    ...    n_blocks=4,
    ...    d_block=5,
    ...    dropout=0.1,
    >>> )
    >>> assert m(x).shape == (batch_size, d_out)
    """

    def __init__(
        self,
        *,
        d_in: int,
        d_out: Optional[int],
        n_blocks: int,
        d_block: int,
        dropout: float,
    ) -> None:
        """
        Args:
            d_in: the input size.
            d_out: the output size.
            n_blocks: the number of blocks.
            d_block: the block width.
            dropout: the dropout rate.
        """
        assert n_blocks > 0
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict(
                        [
                            ('linear', nn.Linear(d_block if i else d_in, d_block)),
                            ('activation', nn.ReLU()),
                            ('dropout', nn.Dropout(dropout)),
                        ]
                    )
                )
                for i in range(n_blocks)
            ]
        )
        """The blocks."""
        self.output = None if d_out is None else nn.Linear(d_block, d_out)
        """The output module."""

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        for block in self.blocks:
            x = block(x)
        if self.output is not None:
            x = self.output(x)
        return x


class ResNet(nn.Module):
    """The ResNet model from Section 3.2 in the paper.

    ```
    ResNet: (in) -> Linear -> Block -> ... -> Block -> Output -> (out)

             |-> BatchNorm -> Linear -> ReLU -> Dropout -> Linear -> Dropout -> |
             |                                                                  |
    Block:  (in) ------------------------------------------------------------> Add -> (out)

    Output: (in) -> BatchNorm -> ReLU -> Linear -> (out)
    ```

    **Shape**

    - Input: `(*, d_in)`
    - Output: `(*, d_out or d_block)`

    **Examples**

    >>> batch_size = 2
    >>> x = torch.randn(batch_size, 2)
    >>> d_out = 1
    >>> m = ResNet(
    ...     d_in=x.shape[1],
    ...     d_out=d_out,
    ...     n_blocks=2,
    ...     d_block=3,
    ...     d_hidden=4,
    ...     d_hidden_multiplier=None,
    ...     dropout1=0.25,
    ...     dropout2=0.0,
    >>> )
    >>> assert m(x).shape == (batch_size, d_out)
    """  # noqa: E501

    def __init__(
        self,
        *,
        d_in: int,
        d_out: Optional[int],
        n_blocks: int,
        d_block: int,
        d_hidden: Optional[int],
        d_hidden_multiplier: Optional[float],
        dropout1: float,
        dropout2: float,
    ) -> None:
        """
        Args:
            d_in: the input size.
            d_out: the output size.
            n_blocks: the number of blocks.
            d_block: the "main" block width (i.e. its input and output size).
            d_hidden: the block's hidden width.
            d_hidden_multipler: the alternative way to set `d_hidden` as
                `int(d_block * d_hidden_multipler)`.
            dropout1: the hidden dropout rate.
            dropout2: the residual dropout rate.
        """
        assert n_blocks > 0
        assert (d_hidden is None) ^ (d_hidden_multiplier is None)
        if d_hidden is None:
            d_hidden = int(d_block * cast(float, d_hidden_multiplier))
        super().__init__()

        self.input_projection = nn.Linear(d_in, d_block)
        """The first linear layer (applied before the main blocks) which
        projects the input from `d_in` to `d_block`."""
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict(
                        [
                            ('normalization', nn.BatchNorm1d(d_block)),
                            ('linear1', nn.Linear(d_block, d_hidden)),
                            ('activation', nn.ReLU()),
                            ('dropout1', nn.Dropout(dropout1)),
                            ('linear2', nn.Linear(d_hidden, d_block)),
                            ('dropout2', nn.Dropout(dropout2)),
                        ]
                    )
                )
                for _ in range(n_blocks)
            ]
        )
        """The blocks."""
        self.output = (
            None
            if d_out is None
            else nn.Sequential(
                OrderedDict(
                    [
                        ('normalization', nn.BatchNorm1d(d_block)),
                        ('activation', nn.ReLU()),
                        ('linear', nn.Linear(d_block, d_out)),
                    ]
                )
            )
        )
        """The output module."""

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        x = self.input_projection(x)
        for block in self.blocks:
            x = x + block(x)
        if self.output is not None:
            x = self.output(x)
        return x


class CLSEmbedding(nn.Module):
    """The [CLS]-token embedding for Transformer-like backbones.

    The module prepends the same trainable token embedding to
    all objects in the batch.

    **Shape**

    - Input: `(batch_size, n_tokens, d_embedding)`
    - Output: `(batch_size, 1 + n_tokens, d_embedding)`
    """

    def __init__(self, d_embedding: int) -> None:
        """
        Args:
            d_embedding: the size of one token embedding
        """
        super().__init__()
        self.weight = Parameter(torch.empty(d_embedding))
        self.reset_parameters()

    @property
    def d_embedding(self) -> int:
        """The embedding size."""
        return self.weight.shape[-1]

    def reset_parameters(self) -> None:
        """Reinitialize all parameters."""
        _init_uniform_rsqrt(self.weight, self.d_embedding)

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        assert x.ndim == 3
        assert x.shape[-1] == self.d_embedding
        return torch.cat([self.weight.expand(len(x), 1, -1), x], dim=1)


class LinearEmbeddings(nn.Module):
    """Linear embeddings for continuous features.

    For the illustration, see `FTTransformer`.

    **Shape**

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`

    **Examples**

    >>> batch_size = 2
    >>> n_cont_features = 3
    >>> x = torch.randn(batch_size, n_cont_features)
    >>> d_embedding = 4
    >>> m = LinearEmbeddings(n_cont_features, d_embedding)
    >>> assert m(x).shape == (batch_size, n_cont_features, d_embedding)
    """

    def __init__(self, n_features: int, d_embedding: int) -> None:
        """
        Args:
            n_features: the number of continous features
            d_embedding: the embedding size
        """
        assert n_features > 0
        assert d_embedding > 0
        super().__init__()

        self.weight = Parameter(torch.empty(n_features, d_embedding))
        """The weight."""
        self.bias = Parameter(torch.empty(n_features, d_embedding))
        """The bias."""
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialize all parameters."""
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                _init_uniform_rsqrt(parameter, self.d_embedding)

    @property
    def n_features(self) -> int:
        """The number of features."""
        return len(self.weight)

    @property
    def d_embedding(self) -> int:
        """The embedding size."""
        return self.weight.shape[1]

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        assert x.ndim == 2
        x = self.weight[None] * x[..., None]
        x = x + self.bias[None]
        return x


class CategoricalFeatureEmbeddings(nn.Module):
    """Embeddings for categorical features.

    For the illustration, see `FTTransformer`.

    **Notes**

    - A cardinality of a categorical feature is the number of distinct values
      that the feature takes
    - A categorical feature must be represented by `int64` from `range(0, cardinality)`

    **Shape**

    - Input: `(*, len(cardinalities))`
    - Output: `(*, len(cardinalities), d_embedding)`

    **Examples**

    >>> cardinalities = [3, 10]
    >>> x = torch.tensor([
    ...     [0, 5],
    ...     [1, 7],
    ...     [0, 2],
    ...     [2, 4]
    ... ])
    >>> batch_size, n_cat_features = x.shape
    >>> d_embedding = 3
    >>> m = CategoricalFeatureEmbeddings(cardinalities, d_embedding, True)
    >>> assert m(x).shape == (batch_size, n_cat_features, d_embedding)
    """

    _category_offsets: Tensor

    def __init__(self, cardinalities: List[int], d_embedding: int, bias: bool) -> None:
        """
        Args:
            cardinalities: the number of distinct values for each feature. For example,
                `cardinalities=[3, 4]` describes two features, where the first
                takes values in the range `[0, 1, 2]` and the second one takes
                values in the range `[0, 1, 2, 3]`.
            d_embedding: the embedding size.
            bias: if `True`, for each feature, a trainable vector is added to the
                embedding regardless of a feature value. For each feature, a separate
                non-shared bias vector is allocated. In the paper, we used `bias=True`.
        """
        super().__init__()
        assert cardinalities
        assert d_embedding > 0

        category_offsets = torch.tensor([0] + cardinalities[:-1]).cumsum(0)
        self.register_buffer('_category_offsets', category_offsets, persistent=False)
        self.embeddings = nn.Embedding(sum(cardinalities), d_embedding)
        """The embeddings."""
        self.bias = (
            Parameter(torch.empty(len(cardinalities), d_embedding)) if bias else None
        )
        """The bias."""
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialize all parameters."""
        for parameter in [self.embeddings.weight, self.bias]:
            if parameter is not None:
                _init_uniform_rsqrt(parameter, self.d_embedding)

    @property
    def n_features(self) -> int:
        """The number of features."""
        return len(self._category_offsets)

    @property
    def d_embedding(self) -> int:
        """The embedding size."""
        return self.embeddings.embedding_dim

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        assert x.ndim == 2
        x = self.embeddings(x + self._category_offsets[None])
        if self.bias is not None:
            x = x + self.bias[None]
        return x


_KV_COMPRESSION_SHARING = Literal['headwise', 'key-value']


class MultiheadAttention(nn.Module):
    """Multihead Attention (self-/cross-) with an optional linear attention.

    - To learn more about Multihead Attention, see
      ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
    - To learn about the linear attention supported by this module, see
      ["Linformer: Self-Attention with Linear Complexity"](https://arxiv.org/abs/2006.04768),

    **Shape**

    - Input:
        - `x_q  ~ (batch_size, n_q_tokes,  d_embedding)`
        - `x_kv ~ (batch_size, n_kv_tokes, d_embedding)`
    - Output: `(batch_size, n_q_tokes, d_embedding)`

    **Examples**

    >>> batch_size, n_tokens, d_embedding = 2, 3, 16
    >>> n_heads = 8
    >>> a = torch.randn(batch_size, n_tokens, d_embedding)
    >>> b = torch.randn(batch_size, n_tokens * 2, d_embedding)
    >>> m = MultiheadAttention(
    ...     d_embedding=d_embedding, n_heads=n_heads, dropout=0.2
    >>> )
    >>>
    >>> # self-attention
    >>> assert m(a, a).shape == a.shape
    >>>
    >>> # cross-attention
    >>> assert m(a, b).shape == a.shape
    >>>
    >>> # Linformer attention
    >>> m = MultiheadAttention(
    ...     d_embedding=d_embedding,
    ...     n_heads=n_heads,
    ...     dropout=0.2,
    ...     n_tokens=n_tokens,
    ...     kv_compression_ratio=0.5,
    ...     kv_compression_sharing='headwise',
    >>> )
    >>> assert m(a, a).shape == a.shape
    """

    def __init__(
        self,
        *,
        d_embedding: int,
        n_heads: int,
        dropout: float,
        # Linformer arguments.
        n_tokens: Optional[int] = None,
        kv_compression_ratio: Optional[float] = None,
        kv_compression_sharing: Optional[_KV_COMPRESSION_SHARING] = None,
    ) -> None:
        """
        Args:
            d_embedding: the embedding size for one token.
                Must be a multiple of `n_heads`.
            n_heads: the number of heads. If greater than 1, then the module will have
                an additional output layer (the so called "mixing" layer).
            dropout: the dropout rate for the attention probability map.
            n_tokens: the number of tokens
                (must be provided if `kv_compression_ratio` is not None)
            kv_compression_ratio: Linformer-style compression rate.
                Must be within the interval `(0.0, 1.0)`.
            kv_compression_sharing: Linformer compression sharing policy.
                Must be provided if `kv_compression_ratio` is not None.
                (non-shared Linformer compression is not supported; the "layerwise"
                sharing policy is not supported).
        """
        if n_heads > 1:
            assert d_embedding % n_heads == 0
        super().__init__()

        self.W_q = nn.Linear(d_embedding, d_embedding)
        """The query projection layer."""
        self.W_k = nn.Linear(d_embedding, d_embedding)
        """The key projection layer."""
        self.W_v = nn.Linear(d_embedding, d_embedding)
        """The value projection layer."""
        self.W_out = nn.Linear(d_embedding, d_embedding) if n_heads > 1 else None
        """The output mixing layer (presented if `n_heads > 1`)."""
        self._n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None
        """The dropout for the attention probability map."""

        if kv_compression_ratio is not None:
            assert n_tokens is not None
            assert kv_compression_sharing in typing.get_args(_KV_COMPRESSION_SHARING)
            assert 0.0 < kv_compression_ratio < 1.0

            def make_kv_compression():
                return nn.Linear(
                    n_tokens, max(int(n_tokens * kv_compression_ratio), 1), bias=False
                )

            self.key_compression = make_kv_compression()
            self.value_compression = (
                make_kv_compression() if kv_compression_sharing == 'headwise' else None
            )
        else:
            assert n_tokens is None
            assert kv_compression_sharing is None
            self.key_compression = None
            self.value_compression = None

        for m in [self.W_q, self.W_k, self.W_v]:
            nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self._n_heads
        return (
            x.reshape(batch_size, n_tokens, self._n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self._n_heads, n_tokens, d_head)
        )

    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tensor:
        """Do the forward pass."""
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        if self.key_compression is not None:
            k = self.key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = (
                self.key_compression
                if self.value_compression is None
                else self.value_compression
            )(v.transpose(1, 2)).transpose(1, 2)

        batch_size = len(q)
        d_head_key = k.shape[-1] // self._n_heads
        d_head_value = v.shape[-1] // self._n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        attention_probs = F.softmax(attention_logits, dim=-1)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)
        x = attention_probs @ self._reshape(v)
        x = (
            x.reshape(batch_size, self._n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self._n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class _ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class FTTransformerBackbone(nn.Module):
    """The backbone of FT-Transformer.

    For the illustration, see `FTTransformer`.

    In fact, it is almost idential to Transformer from the paper
    ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).
    The differences are as follows:
    - the so called "PreNorm" variation is used
      (`norm_first=True` in terms of `torch.nn.TransformerEncoderLayer`)
    - the very first normalization is skipped. This is **CRUCIAL** for FT-Transformer.
    - the ReGLU activation is used in the feed-forward blocks. This is unlikely to be
      crucial, but this is what we used in the paper.

    **Shape**

    - Input: `(batch_size, n_tokens, d_block)`
    - Output: `(batch_size, d_out or d_block)`

    **Examples**

    >>> batch_size = 2
    >>> n_tokens = 3
    >>> d_block = 16
    >>> x = torch.randn(batch_size, n_tokens, d_block)
    >>> d_out = 1
    >>> m = FTTransformerBackbone(
    ...     d_out=d_out,
    ...     n_blocks=2,
    ...     d_block=d_block,
    ...     attention_n_heads=8,
    ...     attention_dropout=0.2,
    ...     ffn_d_hidden=None,
    ...     ffn_d_hidden_multiplier=4 / 3,
    ...     ffn_dropout=0.1,
    ...     residual_dropout=0.0,
    ... )
    >>> assert m(x).shape == (batch_size, d_out)
    """

    def __init__(
        self,
        *,
        d_out: Optional[int],
        n_blocks: int,
        d_block: int,
        attention_n_heads: int,
        attention_dropout: float,
        ffn_d_hidden: Optional[int],
        ffn_d_hidden_multiplier: Optional[float],
        ffn_dropout: float,
        residual_dropout: float,
        n_tokens: Optional[int] = None,
        attention_kv_compression_ratio: Optional[float] = None,
        attention_kv_compression_sharing: Optional[_KV_COMPRESSION_SHARING] = None,
    ):
        """
        Args:
            d_out: the output size.
            n_blocks: the number of blocks.
            d_block: the block width
                (or, equivalently, the embedding size of each feature).
                Must be a multiple of `attention_n_heads`.
            attention_n_heads: the argument for `MultiheadAttention`.
            attention_dropout: the argument for `MultiheadAttention`.
            ffn_d_hidden: the hidden representation size after the activation in the
                feed-forward blocks (or, equivalently, the *input* size of the *second*
                linear layer in the feed-forward blocks). Since `FTTransformerBackbone`
                uses ReGLU activation function, the *output* size of the *first*
                linear layer will be `2 * ffn_d_hidden`.
            ffn_d_hidden_multiplier: the alternative way to set `ffn_d_hidden` as
                `int(d_block * ffn_d_hidden_multiplier)`
            ffn_dropout: the dropout rate for the hidden representation
                in the feed-forward blocks.
            residual_dropout: the dropout rate for all residual branches.
            n_tokens: the argument for `MultiheadAttention`.
            attention_kv_compression_ratio: the argument for `MultiheadAttention`.
                Use this option with caution:
                - it can affect task performance in an unpredictable way
                - it can make things *slower* when the number of features
                  is not large enough
            attention_kv_compression_sharing: the argument for `MultiheadAttention`.
        """
        if attention_kv_compression_sharing is not None:
            assert attention_kv_compression_sharing in typing.get_args(
                _KV_COMPRESSION_SHARING
            )
        assert (ffn_d_hidden is None) ^ (ffn_d_hidden_multiplier is None)
        if ffn_d_hidden is None:
            ffn_d_hidden = int(d_block * cast(float, ffn_d_hidden_multiplier))
        super().__init__()

        self.cls_embedding = CLSEmbedding(d_block)
        """The [CLS]-token embedding."""

        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        # >>> attention
                        'attention': MultiheadAttention(
                            d_embedding=d_block,
                            n_heads=attention_n_heads,
                            dropout=attention_dropout,
                            n_tokens=n_tokens,
                            kv_compression_ratio=attention_kv_compression_ratio,
                            kv_compression_sharing=attention_kv_compression_sharing,
                        ),
                        'attention_residual_dropout': nn.Dropout(residual_dropout),
                        # >>> feed-forward
                        'ffn_normalization': nn.LayerNorm(d_block),
                        'ffn': nn.Sequential(
                            OrderedDict(
                                [
                                    # Multiplying dimension by 2 to compensate for
                                    # ReGLU which (internally) divides dimension by 2.
                                    ('linear1', nn.Linear(d_block, ffn_d_hidden * 2)),
                                    ('activation', _ReGLU()),
                                    ('dropout', nn.Dropout(ffn_dropout)),
                                    ('linear2', nn.Linear(ffn_d_hidden, d_block)),
                                ]
                            )
                        ),
                        'ffn_residual_dropout': nn.Dropout(residual_dropout),
                        # >>> output (for hooks-based introspection)
                        'output': nn.Identity(),
                        # >>> the very first normalization
                        **(
                            {}
                            if layer_idx == 0
                            else {'attention_normalization': nn.LayerNorm(d_block)}
                        ),
                    }
                )
                for layer_idx in range(n_blocks)
            ]
        )
        """The blocks."""
        self.output = (
            None
            if d_out is None
            else nn.Sequential(
                OrderedDict(
                    [
                        ('normalization', nn.LayerNorm(d_block)),
                        ('activation', nn.ReLU()),
                        ('linear', nn.Linear(d_block, d_out)),
                    ]
                )
            )
        )
        """The output module."""

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        assert x.ndim == 3

        x = self.cls_embedding(x)

        n_blocks = len(self.blocks)
        for i_block, block in enumerate(self.blocks):
            block = cast(nn.ModuleDict, block)

            x_identity = x
            if 'attention_normalization' in block:
                x = block['attention_normalization'](x)
            x = block['attention'](x[:, :1] if i_block + 1 == n_blocks else x, x)
            x = block['attention_residual_dropout'](x)
            x = x_identity + x

            x_identity = x
            x = block['ffn_normalization'](x)
            x = block['ffn'](x)
            x = block['ffn_residual_dropout'](x)
            x = x_identity + x

            x = block['output'](x)

        x = x[:, 0]  # The representation of [CLS]-token.

        if self.output is not None:
            x = self.output(x)
        return x


class FTTransformer(nn.Module):
    """The FT-Transformer model from Section 3.3 in the paper.

    <img src="ft-transformer-overview.png" width=100%>

    .. note::

        We should admit that "Feature Tokenizer" is a bad and misleading name,
        which misuses the term "token". A better name would be "Feature Embeddings".

    <img src="ft-transformer-details.png" width=100%>

    The default hyperparameters can be obtained with `FTTransformer.get_default_kwargs`.

    **Shape**

    - Input:
        - continuous features: `x_cont ~ (batch_size, n_cont_features)`
        - categorical features: `x_cat ~ (batch_size, len(cat_cardinalities))`
    - Output: `(batch_size, d_out or d_block)`

    **Examples**

    >>> batch_size = 2
    >>> n_cont_feaatures = 3
    >>> cardinalities = [3, 4]
    >>> x_cont = torch.randn(batch_size, n_cont_feaatures)
    >>> x_cat = torch.column_stack([
    ...     torch.randint(0, c, (batch_size,))
    ...     for c in cardinalities
    ... ])
    >>> d_out = 1
    >>> m = FTTransformer(
    ...     n_cont_features=n_cont_feaatures,
    ...     cat_cardinalities=cardinalities,
    ...     d_out=d_out,
    ...     n_blocks=2,
    ...     d_block=16,
    ...     attention_n_heads=8,
    ...     attention_dropout=0.2,
    ...     ffn_d_hidden=None,
    ...     ffn_d_hidden_multiplier=4 / 3,
    ...     ffn_dropout=0.1,
    ...     residual_dropout=0.0,
    ... )
    >>> assert m(x_cont, x_cat).shape == (batch_size, d_out)
    """

    def __init__(
        self,
        *,
        n_cont_features: int,
        cat_cardinalities: List[int],
        _is_default: bool = False,
        **backbone_kwargs,
    ) -> None:
        """
        Args:
            n_cont_features: the number of continuous features
            cat_cardinalities: the cardinalities of categorical features (see
                `CategoricalFeatureEmbeddings` for details). Pass en empty list
                if there are no categorical features.
            _is_default: this is a technical argument, don't set it manually.
            backbone_kwargs: the keyword arguments for the `FTTransformerBackbone`
        """
        assert n_cont_features >= 0
        assert all(x > 0 for x in cat_cardinalities)
        assert n_cont_features > 0 or cat_cardinalities
        super().__init__()

        d_block: int = backbone_kwargs['d_block']
        # >>> Feature embeddings (see Figure 2a in the paper).
        self.cont_embeddings = (
            LinearEmbeddings(n_cont_features, d_block) if n_cont_features > 0 else None
        )
        """The embeddings for continuous features."""
        self.cat_embeddings = (
            CategoricalFeatureEmbeddings(cat_cardinalities, d_block, True)
            if cat_cardinalities
            else None
        )
        """The embeddings for categorical features."""
        # >>>
        self.backbone = FTTransformerBackbone(**backbone_kwargs)
        """The backbone."""
        self._is_default = _is_default

    @classmethod
    def get_default_kwargs(cls, n_blocks: int = 3) -> Dict[str, Any]:
        """Get the default hyperparameters.

        Args:
            n_blocks: the number of blocks. The supported values are in `range(1, 7)`.
        Returns:
            the default keyword arguments for the constructor

        **Examples**

        >>> m = FTTransformer(
        ...     n_cont_features=3,
        ...     cat_cardinalities=[4, 5],
        ...     d_out=1,
        ...     **FTTransformer.get_default_kwargs()
        ... )
        """
        assert (
            1 <= n_blocks <= 6
        ), 'We offer default configurations only for `n_blocks in range(1, 7)`'
        return {
            'n_blocks': n_blocks,
            'd_block': [96, 128, 192, 256, 320, 384][n_blocks - 1],
            'attention_n_heads': 8,
            'attention_dropout': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35][n_blocks - 1],
            # Because of the ReGLU activation used by FT-Transformer,
            # 4 / 3 results in roughly the same number of parameters as 2.0
            # would with simple element-wise activations (e.g. ReLU).
            'ffn_d_hidden': None,
            'ffn_d_hidden_multiplier': 4 / 3,
            'ffn_dropout': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25][n_blocks - 1],
            'residual_dropout': 0.0,
            '_is_default': True,
        }

    def make_parameter_groups(self) -> List[Dict[str, Any]]:
        """Make parameter groups for optimizers.

        The difference with calling this method instead of
        `.parameters()` is that this method always sets `weight_decay=0.0`
        for some of the parameters.

        **Examples**

        >>> m = FTTransformer(
        ...     n_cont_features=2,
        ...     cat_cardinalities=[3, 4],
        ...     d_out=5,
        ...     **FTTransformer.get_default_kwargs(),
        ... )
        >>> optimizer = torch.optim.AdamW(
        ...     m.make_parameter_groups(),
        ...     lr=1e-4,
        ...     weight_decay=1e-5,
        ... )
        """
        main_group: Dict[str, Any] = {'params': []}
        zero_wd_group: Dict[str, Any] = {'params': [], 'weight_decay': 0.0}

        zero_wd_subnames = ['normalization', '.bias']
        for modulename in ['cont_embeddings', 'cat_embeddings', 'cls_embedding']:
            if getattr(self, modulename, None) is not None:
                zero_wd_subnames.append(modulename)
        # Check that there are no typos in the above list.
        for subname in zero_wd_subnames:
            assert any(
                subname in name for name, _ in self.named_parameters()
            ), _INTERNAL_ERROR_MESSAGE

        for name, parameter in self.named_parameters():
            zero_wd_condition = any(subname in name for subname in zero_wd_subnames)
            (zero_wd_group if zero_wd_condition else main_group)['params'].append(
                parameter
            )
        return [main_group, zero_wd_group]

    def make_default_optimizer(self) -> torch.optim.AdamW:
        """Create the "default" `torch.nn.AdamW` suitable for the *default* FT-Transformer.

        Returns:
            optimizer

        **Examples**

        >>> m = FTTransformer(
        ...     n_cont_features=2,
        ...     cat_cardinalities=[3, 4],
        ...     d_out=5,
        ...     **FTTransformer.get_default_kwargs(),
        ... )
        >>> optimizer = m.make_default_optimizer()
        """  # noqa: E501
        if not self._is_default:
            warnings.warn(
                'The default opimizer is supposed to be used in a combination'
                ' with the default FT-Transformer.'
            )
        return torch.optim.AdamW(
            self.make_parameter_groups(), lr=1e-4, weight_decay=1e-5
        )

    _FORWARD_BAD_ARGS_MESSAGE = (
        'Based on the arguments passed to the constructor of FT-Transformer, {}'
    )

    def forward(self, x_cont: Optional[Tensor], x_cat: Optional[Tensor]) -> Tensor:
        """Do the forward pass.

        Args:
            x_cont: the continuous features.
            x_cat: the categorical features.
        """
        assert x_cont is not None or x_cat is not None

        x_embeddings = []
        for argname, module in [
            ('x_cont', self.cont_embeddings),
            ('x_cat', self.cat_embeddings),
        ]:
            argvalue = locals()[argname]
            if module is None:
                assert argvalue is None, FTTransformer._FORWARD_BAD_ARGS_MESSAGE.format(
                    f'{argname} must be None'
                )
            else:
                assert (
                    argvalue is not None
                ), FTTransformer._FORWARD_BAD_ARGS_MESSAGE.format(
                    f'{argname} must not be None'
                )
                x_embeddings.append(module(argvalue))
        assert x_embeddings
        x = torch.cat(x_embeddings, dim=1)
        x = self.backbone(x)
        return x
