# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed — Stiefel constraint moved from `beta` to `energy`

`Model(..., energy_manifold="stiefel")` constrains the feature-side energy
matrix `E ∈ R^{k×i}` to have orthonormal rows (`E Eᵀ = I_k`), stored as
`energy_T` with shape `(features, k)` on geoopt's column-orthonormal Stiefel
manifold. `beta` is always Euclidean. The forward pass uses `E = energy_T.T`,
so `sigmoid(-(beta @ E))` is unchanged in shape and semantics.

- **Breaking:** `beta_manifold` was removed; pass `energy_manifold="stiefel"`.
- Requires `features >= latent_dim` (was `samples >= latent_dim`).
- `Model.total_params` reduces the **energy** term to `k·i − k(k+1)/2`.
- Riemannian default optimizers and warnings unchanged in spirit.

### Added — Stiefel-manifold constraint on `energy` (optional)

- Requires the optional `geoopt` dependency: `pip install sigmoid-py[geometry]`.
- When `energy_manifold="stiefel"` is set, the default optimizer switches from
  `torch.optim.Adam` to `geoopt.optim.RiemannianAdam`. `RiemannianSGD` is also
  available (`optimizer="sgd"`). A non-Riemannian optimizer raises a
  `RuntimeWarning` (the constraint would drift).
- `AdamW` is **not** appropriate for manifold-constrained parameters.
- `Model.energy_matrix()` returns the paper-shaped `(k, features)` view.
- New optional-dep group: `[project.optional-dependencies] geometry = ["geoopt>=0.5"]`.
- Ten new tests in `tests/test_stiefel.py`, all guarded by
  `pytest.importorskip("geoopt")` so the suite still runs without geoopt.

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
  now **`torch.optim.Adam(lr=nu)`** with `nu=0.01` (was `SGD(lr=0.001)`).
  Rationale: on the included benchmark (`examples/bench_optimizers.py`,
  200×60 binary data, k=4, 5 seeds) Adam at `lr=1e-2` finds a slightly lower
  final NLL and is robust to `nu`, while SGD at the previous default
  `lr=1e-3` is ~4× slower to converge and SGD at `lr=1e-1` diverges.
  A user-supplied `optimizer=` may be passed to `fit()` to use SGD, AdamW,
  schedulers, etc. **Note for throughput-conscious users:** plain
  `SGD(lr=nu)` reaches the same plateau in roughly half the wall-time
  (no momentum/variance state to update each step) and is mathematically
  equivalent to the original hand-rolled SiGMoiD update — pass it explicitly
  if you have a known-good `nu`.
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
