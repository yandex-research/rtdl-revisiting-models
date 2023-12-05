# Python package <!-- omit in toc -->

> [!NOTE]
> See also [RTDL](https://github.com/yandex-research/rtdl)
> -- other projects on tabular deep learning.

This package provides the officially recommended implementation
of the paper "Revisiting Deep Learning Models for Tabular Data".

<details>
<summary><i>This package VS The original implementation</i></summary>

"Original implementation" is the code in `bin/` and `lib/`
used to obtain numbers reported in the paper.

- **This package is recommended over the original implementation**:
  the package is significanty simpler
  while being fully consistent with the original code.
- Strictly speaking, there are tiny technical divergences from the original code,
  however, they don't affect anything important.
  Just in case, they are marked
  with `# NOTE[DIFF]` comments in the source code of this package.
  Any divergence from the original implementation without the `# NOTE[DIFF]` comment
  is considered to be a bug.

</details>

---

- [Installation](#installation)
- [Usage](#usage)
- [End-to-end examples](#end-to-end-examples)
- [Practical notes](#practical-notes)
- [API](#api)
- [Development](#development)

# Installation

*(RTDL ~ **R**esearch on **T**abular **D**eep **L**earning)*

```
pip install rtdl_revisiting_models
```

# Usage

> [!IMPORTANT]
> It is recommended to first read the TL;DR of the paper:
> [link](../README.md#tldr)

The package provides the following PyTorch modules:
- `MLP`
- `ResNet`
- `FTTransformer` (proposed in the paper)
- Technical modules used by `FTTransformer` (feature embeddings, attention, etc.).

The common setup for all examples:
a batch of objects with continuous and categorical features:

<!-- test main -->
```python
# NOTE: all code snippets can be copied and executed as-is.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from rtdl_revisiting_models import MLP, ResNet, FTTransformer

batch_size = 2

# Continuous features.
n_cont_features = 3
x_cont = torch.randn(batch_size, n_cont_features)

# Categorical features.
cat_cardinalities = [
    4,  # Allowed values: [0, 1, 2, 3].
    7,  # Allowed values: [0, 1, 2, 3, 4, 5, 6].
]
n_cat_features = len(cat_cardinalities)
x_cat = torch.column_stack([
    torch.randint(0, c, (batch_size,))
    for c in cat_cardinalities
])
assert x_cat.dtype == torch.int64
assert x_cat.shape == (batch_size, n_cat_features)

# MLP-like models (e.g. MLP and ResNet) require
# categorical features to be encoded as continuous features.
# One way to achieve that is the one-hot encoding
# (for features with high cardinality, embeddings can be a better choice).
x_cat_ohe = [
    F.one_hot(cat_column, c)
    for cat_column, c in zip(x_cat.T, cat_cardinalities)
]
x = torch.column_stack([x_cont] + x_cat_ohe)
assert x.shape == (
    batch_size, n_cont_features + sum(cat_cardinalities)
)
```

## MLP <!-- omit in toc -->

*(Decribed in Section 3.1 in the paper)*

```
MLP:   (in) -> Block  -> ...  -> Block   -> [Output ->] (out)
Block: (in) -> Linear -> ReLU -> Dropout -> (out)
Output = Linear
```

<!-- test main _ -->
```python
d_out = 1  # For example, a single regression task.
model = MLP(
    d_in=n_cont_features + sum(cat_cardinalities),
    d_out=d_out,
    n_blocks=2,
    d_block=384,
    dropout=0.1,
)
y_pred = model(x)
assert y_pred.shape == (batch_size, d_out)
```

## ResNet <!-- omit in toc -->

*(Decribed in Section 3.2 in the paper)*

```
ResNet: (in) -> Linear -> Block -> ... -> Block -> [Output ->] (out)

            |-> BatchNorm -> Linear -> ReLU -> Dropout -> Linear -> Dropout -> |
            |                                                                  |
Block:  (in) ------------------------------------------------------------> Add -> (out)

Output: (in) -> BatchNorm -> ReLU -> Linear -> (out)
```

<!-- test main _ -->
```python
d_out = 1  # For example, a single regression task.
model = ResNet(
    d_in=n_cont_features + sum(cat_cardinalities),
    d_out=d_out,
    n_blocks=2,
    d_block=192,
    d_hidden=None,
    d_hidden_multiplier=2.0,
    dropout1=0.15,
    dropout2=0.0,
)
y_pred = model(x)
assert y_pred.shape == (batch_size, d_out)
```

## FT-Transformer <!-- omit in toc -->

*(Decribed in Section 3.3 in the paper)*

> [!IMPORTANT]
> The backbone of FT-Transformer has a small, but *crucial* technical difference
> from the original Transformer model from the "Attention is all you need" paper.
> For details, see the docstrings for `FTTransformerBackbone` in the source code.

<img src="ft-transformer-overview.png" width=70%>

<img src="ft-transformer-details.png" width=70%>

<!-- test main _ -->
```python
d_out = 1  # For example, a single regression task.

# FT-Transformer expects continuous and categorical
# features to be passed separately.
model = FTTransformer(
    n_cont_features=n_cont_features,
    cat_cardinalities=cat_cardinalities,
    d_out=d_out,
    n_blocks=3,
    d_block=192,
    attention_n_heads=8,
    attention_dropout=0.2,
    ffn_d_hidden=None,
    ffn_d_hidden_multiplier=4 / 3,
    ffn_dropout=0.1,
    residual_dropout=0.0,
)
assert model(x_cont, x_cat).shape == (batch_size, d_out)

# In the paper, some of FT-Transformer's parameters
# were protected from the weight decay regularization.
# There is a special method for doing that:
optimizer = torch.optim.AdamW(
    # Instead of model.parameters(),
    model.make_parameter_groups(),
    lr=1e-4,
    weight_decay=1e-5,
)
```

For a quick start, there is a default configuration:

<!-- test main _ -->
```python
d_out = 1
default_kwargs = FTTransformer.get_default_kwargs()
model = FTTransformer(
    n_cont_features=n_cont_features,
    cat_cardinalities=cat_cardinalities,
    d_out=d_out,
    **default_kwargs,
)
optimizer = model.make_default_optimizer()
assert model(x_cont, x_cat).shape == (batch_size, d_out)
```

When the number of features is large and the quadratic complexity of the attention
mechanism becomes an issue, the [Linformer](https://arxiv.org/abs/2006.04768) attention
can be used to accelerate FT-Transformer.
Note that it can make things *slower* when the number of features is not large enough.

> [!NOTE]
> The influence of Linformer attention on the task performance
of FT-Transformer is not well-studied.

<!-- test main _ -->
```python
x_cont = torch.randn(batch_size, 1024)  # Many features
d_out = 1
model = FTTransformer(
    n_cont_features=x_cont.shape[1],
    cat_cardinalities=cat_cardinalities,
    d_out=d_out,
    n_blocks=3,
    d_block=192,
    attention_n_heads=8,
    attention_dropout=0.2,
    ffn_d_hidden=None,
    ffn_d_hidden_multiplier=4 / 3,
    ffn_dropout=0.1,
    residual_dropout=0.0,
    linformer_kv_compression_ratio=0.2,           # <---
    linformer_kv_compression_sharing='headwise',  # <---
)
assert model(x_cont, x_cat).shape == (batch_size, d_out)
```

# End-to-end examples

See [this Jupyter notebook](./example.ipynb) (Colab link inside).

# Practical notes

**Models**

- MLP is a simple lightweight baseline,
  which is great for implementing the first version of a tabular DL pipeline.
- Strictly speaking, ResNet is better (and slower) than MLP on average.
  In practice, the gap between ResNet and MLP depends on a dataset,
  and ResNet performing similarly to MLP is not an anomaly.
- FT-Transformer is more powerful than MLP and Resnet and is slower than them.
  With FT-Transformer, it is usually easy to obtain decent results
  with non-incremental improvements over MLP/ResNet.

**Hyperparameters**

> [!NOTE]
> It is possible to explore tuned hyperparameters
> for the models and datasets used in the paper as explained here:
> [here](../README.md#how-to-explore-metrics-and-hyperparameters).

- `FTTransformer` has a default configuration for a quick start.
- In the paper, for hyperparameter tuning, typically, the
  [TPE sampler from Optuna](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
  was used with `study.optimize(..., n_trials=100)`.
- The hyperparamer tuning spaces can be found in the appendix of the paper
  and in the `output/**/*tuning.toml` files.

# API

To explore the available API and docstrings, open the source file and:
- on GitHub, use the Symbols panel
- in VSCode, use the [Outline view](https://code.visualstudio.com/docs/getstarted/userinterface#_outline-view)
- check the `__all__` variable

# Development

<details>

Set up the environment (replace `micromamba` with `conda` or `mamba` if needed):
```
micromamba create -f environment-package.yaml
```

Check out the available commands in the [Makefile](./Makefile).
In particular, use this command before committing:
```
make pre-commit
```

Publish the package to PyPI (requires PyPI account & configuration):
```
flit publish
```
</details>
