# SiGMoiD

SiGMoiD is a statistical approach to modelling high-dimensional binary data inspired by statistical physics. This package is based on the paper *"SiGMoiD: A super-statistical generative model for binary data"*. Please cite the following if you use this code in your work:

> Zhao X, Plata G, Dixit PD (2021) SiGMoiD: A super-statistical generative model for binary data. *PLOS Computational Biology* 17(8): e1009275. https://doi.org/10.1371/journal.pcbi.1009275

In this implementation, we leverage PyTorch for fast model fitting and inference. We also provide a model-selection framework based on Akaike Information Criterion (AIC).

Please reach out if you have any questions or suggestions!

## Installation

Via pip:

```bash
pip install git+https://github.com/GecklesTheClown/SiGMoiD.git
```

Via uv:

```bash
uv add git+https://github.com/GecklesTheClown/SiGMoiD.git
```

PyPI (coming soon):

```bash
pip install sigmoid-py
```

## GPU Support

This package supports GPU acceleration via PyTorch. If you have a compatible GPU and an appropriate PyTorch build installed, the package will automatically use it for model fitting and inference. By default, the package only declares the CPU PyTorch wheel as a dependency.

For GPU acceleration, install PyTorch with CUDA support following the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).

## Quick Start

```python
import pandas as pd
from sigmoid import Selector

# Load your binary data into a pandas DataFrame (or numpy array of {0,1})
data = pd.read_csv("your_binary_data.csv").to_numpy()

# Initialize the model selector
selector = Selector(data, seed=42)
# Select the best model based on AIC
selector.fit(k=range(1, 21), repeats=10)
model = selector.optimal

# Draw samples from the optimal fitted model
samples = model.draw_samples(n_samples=1000, seed=42)
```

`Model` and `Selector` are also available from the top-level `sigmoid` namespace:

```python
from sigmoid import Model, Selector
```

## PyTorch Interop

`Model` is a `torch.nn.Module`, so it composes with the rest of the PyTorch
ecosystem. You can plug in any optimizer, attach an LR scheduler, persist
weights with `state_dict`, move to GPU with `.to(...)`, etc.

```python
import torch
from sigmoid import Model

model = Model(data, latent_dim=5)

# Bring your own optimizer (Adam, AdamW, RMSprop, ...).
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
model.fit(its=500, optimizer=opt, seed=42)

# Standard nn.Module persistence
torch.save(model.state_dict(), "sigmoid_weights.pt")
```

If `optimizer` is omitted, `fit` defaults to `torch.optim.SGD(lr=nu)`, which
matches the original SiGMoiD update rule.

## Model Selection

`Selector.fit` trains `len(k) * repeats` candidate models and tracks each one in
`selector.trace` as `(latent_dim, aic, seed)` tuples. If pandas is installed you
can call `selector.trace_df()` for a tabular view. Pass `keep_all=True` to
retain every candidate in `selector.candidates` (keyed by
`(latent_dim, seed)`); by default only the running best is retained to save
memory.

## Reproducibility

Set the `seed` argument on `Selector` or `Model` for deterministic behaviour. Each candidate model trained inside `Selector.fit` is seeded from a per-candidate seed derived from the `Selector`'s master seed.

## Limitations

- Input data must be `{0, 1}`-valued. Any numpy dtype is accepted; values are cast to float internally.
- Memory scales as `O(s*k + k*i)` where `s` is the number of samples, `i` the number of features, and `k` the latent dimension.
- Model selection currently uses AIC only. BIC and cross-validation are on the roadmap.

## Roadmap

- More model selection criteria (e.g. BIC, cross-validation with different metrics).
- Improved model selection computational efficiency.
- Adaptive learning-rate schedulers.

## Citing

```bibtex
@article{zhao2021sigmoid,
  title   = {SiGMoiD: A super-statistical generative model for binary data},
  author  = {Zhao, Xiaochuan and Plata, German and Dixit, Purushottam D.},
  journal = {PLOS Computational Biology},
  volume  = {17},
  number  = {8},
  pages   = {e1009275},
  year    = {2021},
  doi     = {10.1371/journal.pcbi.1009275}
}
```
