# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Note — the SiGMoiD probability function (READ ME if touching `Model.forward`)

The paper defines the per-cell probability as

```
              exp(-Σ_k β_sk E_ki)
    p_si  =  ─────────────────────
             1 + exp(-Σ_k β_sk E_ki)
```

Let `z = β @ E` (so `z_si = Σ_k β_sk E_ki`). Then:

```
    p = exp(-z) / (1 + exp(-z))            (paper)
      = 1 / (1 + exp(z))                   (multiply top & bottom by exp(z))
      = sigmoid(-z)                        (definition of sigmoid)
      = 1 - sigmoid(z)                     (sigmoid identity)
```

So **the implementation must use `torch.sigmoid(-(beta @ energy))`**, NOT
`torch.sigmoid(beta @ energy)`. The two differ by `1 - p`, i.e. they swap the
roles of 0 and 1 in the data, which silently breaks training. The previous
hand-rolled `_sigmoid_transform(x)` was a numerically-stable form of
`sigmoid(-x)` (both of its `torch.where` branches simplify to `1/(1+e^x)`); it
was removed in favour of `torch.sigmoid(-z)`, which is mathematically and
numerically equivalent (both use the same overflow-avoiding branch trick
internally).

A unit test (`test_training_actually_minimizes_nll`) guards against an
inverted forward by asserting NLL strictly decreases over training.

### Changed (breaking)
- `Model` is now an `nn.Module`. `beta` and `energy` are `nn.Parameter`s,
  so `model.parameters()`, `model.to(device)`, `model.state_dict()`, and the
  rest of the PyTorch ecosystem (custom optimizers, schedulers, mixed precision,
  `nn.DataParallel`, …) work out of the box.
- The training method is now `Model.fit(...)` instead of `Model.train(...)`.
  This frees `nn.Module.train(mode=True)` for its standard purpose of toggling
  train/eval mode, which downstream PyTorch packages rely on.
- The hand-rolled gradient loop has been replaced with a standard
  `loss.backward(); optimizer.step()` autograd loop. The default optimizer is
  `torch.optim.SGD(lr=nu)`, which produces parameter updates identical to the
  original SiGMoiD update down to floating-point noise (~1e-8 max diff after
  500 iterations on a toy dataset). A user-supplied `optimizer=` may be passed
  to `fit()` to use Adam, AdamW, schedulers, etc.
- The probability function ``sigmoid(-(beta @ energy))`` from the paper is
  preserved verbatim. The previous custom ``_sigmoid_transform`` helper has
  been removed in favour of ``torch.sigmoid(-z)``, which is identical and
  numerically stable.

### Fixed
- `Selector.fit` now correctly forwards its `gpu` argument to `Model.fit`.
- `Model` accepts integer-dtype numpy input (previously caused a BCE failure).
- `Model.draw_samples` no longer mutates global NumPy RNG state and uses
  explicit tensor → numpy conversion.
- Removed the redundant `k` keyword from `Model.train` that silently overrode
  the latent dimension passed to `__init__`.
- Cleaned up docstring typos.

### Added
- Top-level re-exports: `from sigmoid import Model, Selector`, plus `__version__`.
- `Model.fit` accepts an `optimizer=` argument (any `torch.optim.Optimizer`).
- `Model.fit` returns `self` for chaining and supports optional
  `track_loss` and `verbose` flags. A `loss_history` attribute records
  per-iteration negative log-likelihood when enabled.
- `Selector.fit` returns `self` and supports `keep_all=True` to retain every
  trained candidate in `Selector.candidates`.
- `Selector.trace_df()` helper for tabular inspection (lazy pandas import).
- Type hints on the public API (the package already shipped `py.typed`).
- Test suite under `tests/` covering the public interface.
- GitHub Actions CI running ruff + pytest on Python 3.10 / 3.11 / 3.12.
- Ruff and pytest configuration in `pyproject.toml`.
- `dev` and `pandas` optional-dependency groups.

### Changed
- Distribution name set to `sigmoid-py` (import name remains `sigmoid`).
- Filled in real package metadata (description, license, classifiers, URLs).
- Loosened `requires-python` from `>=3.12` to `>=3.10`.
- Dropped unused `torchvision` dependency.
- `verbose` output now goes through `logging` instead of `print`.
- Removed the empty `utils.py` placeholder.

## [0.1.0]

- Initial release.
