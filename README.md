# SiGMoiD

SiGMoiD is a statistical approach to modelling high-dimensional binary data inspired by statistical physics. This package is based on the paper *"SiGMoiD: A super-statistical generative model for binary data"*. Please cite the following if you use this code in your work:

> Zhao X, Plata G, Dixit PD (2021) SiGMoiD: A super-statistical generative model for binary data. *PLOS Computational Biology* 17(8): e1009275. https://doi.org/10.1371/journal.pcbi.1009275

In this implementation, we leverage PyTorch for fast model fitting and inference. We also provide a model-selection framework based on Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC).

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
# Select the best model (AIC by default; pass criterion="bic" for BIC)
selector.fit(k=range(1, 21), repeats=10)
# selector.fit(k=range(1, 21), repeats=10, criterion="bic")
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

# Bring your own optimizer (Adam, AdamW, RMSprop, ...) or scheduler.
opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
model.fit(its=500, optimizer=opt, seed=42)

# Standard nn.Module persistence
torch.save(model.state_dict(), "sigmoid_weights.pt")
```

### Default optimizer

If `optimizer` is omitted, `fit` uses `torch.optim.Adam(lr=nu)` (with
`nu=0.01` by default). Adam is the default because it is robust to the choice
of `nu` and tends to find a slightly lower NLL on typical SiGMoiD problems.

**Want faster training?** Plain `SGD(lr=nu)` reaches the same loss plateau in
roughly half the wall-time on the included benchmark
(`examples/bench_optimizers.py`) — there's no momentum/variance state to
update each step. The catch: SGD is brittle to `nu`. Too high a value
(`>= 0.1` on a typical problem) diverges. If you have a known-good `nu` and
care about throughput, pass it in:

```python
opt = torch.optim.SGD(model.parameters(), lr=1e-2)
model.fit(its=500, optimizer=opt, seed=42)
```

`SGD(lr=nu)` is also mathematically equivalent to the original hand-rolled
SiGMoiD update (down to floating-point noise).

### Optional CUDA speed-ups (`bf16`, `torch.compile`)

For large fits on a recent NVIDIA GPU, you can opt into bfloat16 autocast and/or
`torch.compile` without changing defaults for everyone else:

```python
# bf16: faster matmuls; weights/optimizer stay float32
model.fit(its=2000, gpu=True, bf16=True, seed=42)

# torch.compile: fuses the forward graph (warmup on first steps)
model.fit(its=2000, gpu=True, compile_model=True, seed=42)

# Stiefel + geoopt: same flags; retractions stay full precision
model = Model(data, latent_dim=5, beta_manifold="stiefel")
model.fit(its=2000, gpu=True, bf16=True, compile_model=True, seed=42)
```

`bf16=True` is ignored on CPU (with a warning). `Selector.fit` accepts the
same `bf16` and `compile_model` keyword arguments.

## Constrained parameter learning (Stiefel manifold)

The `beta @ energy` factorization has a rotational gauge ambiguity: any
invertible `R` gives `(beta R) @ (R⁻¹ energy)` with identical likelihood. You
can remove this ambiguity — at no cost to expressivity — by constraining `beta`
to the **Stiefel manifold** `St(s, k) = {B ∈ R^{s×k} : Bᵀ B = I_k}`, i.e.
orthonormal columns. Any unconstrained `beta` admits a QR decomposition
`beta = Q R`, so `beta @ energy = Q @ (R @ energy)` with `Q` on the manifold
and the rotation absorbed into `energy` (which is left Euclidean on purpose —
constraining both would over-constrain the span).

Requires the optional `geoopt` dependency:

```bash
pip install sigmoid-py[geometry]
```

Usage:

```python
from sigmoid import Model

# beta is constrained to have orthonormal columns.
# Default optimizer switches to geoopt.optim.RiemannianAdam, which applies
# the Stiefel retraction after each Adam step so the constraint holds exactly.
model = Model(data, latent_dim=5, beta_manifold="stiefel")
model.fit(its=500, nu=5e-2, seed=42)

# `total_params` (and therefore AIC) automatically uses the reduced Stiefel
# DOF (s*k - k(k+1)/2) so model selection is fair vs. unconstrained fits.
```

Notes:

- Requires `samples >= latent_dim`.
- Use `geoopt.optim.RiemannianAdam` or `RiemannianSGD`. Supplying a plain
  `torch.optim.Adam`/`SGD` raises a `RuntimeWarning` — the step is Euclidean,
  so the constraint drifts.
- **Do not** use `AdamW` with manifold parameters: its decoupled weight decay
  shrinks towards zero, leaving the manifold.

## Model Selection

`Selector.fit` trains `len(k) * repeats` candidate models and tracks each one in
`selector.trace` as `(latent_dim, score, seed)` tuples, where `score` is AIC or
BIC depending on the `criterion` passed to `fit`. If pandas is installed you
can call `selector.trace_df()` for a tabular view. Pass `keep_all=True` to
retain every candidate in `selector.candidates` (keyed by
`(latent_dim, seed)`); by default only the running best is retained to save
memory.

## Reproducibility

Set the `seed` argument on `Selector` or `Model` for deterministic behaviour. Each candidate model trained inside `Selector.fit` is seeded from a per-candidate seed derived from the `Selector`'s master seed.

## Limitations

- Input data must be `{0, 1}`-valued. Any numpy dtype is accepted; values are cast to float internally.
- Memory scales as `O(s*k + k*i)` where `s` is the number of samples, `i` the number of features, and `k` the latent dimension.
- Model selection supports AIC and BIC; cross-validation is on the roadmap.

## Roadmap

- More model selection criteria (e.g. cross-validation with different metrics).
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
