"""Tests for PyTorch orthogonal-parametrized energy."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from torch.nn.utils import parametrize

from sigmoid import Model


def _toy_data(seed: int = 0, shape=(60, 16)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.binomial(1, 0.4, size=shape).astype(np.float32)


def _energy_row_orthonormality_error(energy_ki: torch.Tensor) -> float:
    """Max abs entry of ``E @ E^T - I`` for ``E`` of shape ``(k, features)``."""
    with torch.no_grad():
        k = energy_ki.shape[0]
        err = energy_ki @ energy_ki.T - torch.eye(k, device=energy_ki.device, dtype=energy_ki.dtype)
    return float(err.abs().max())


def test_orthogonal_energy_init_is_parametrized_and_orthonormal():
    model = Model(_toy_data(), latent_dim=4, energy_manifold="orthogonal")
    assert parametrize.is_parametrized(model, "energy_T")
    assert model.energy_T.shape == (16, 4)
    e = model.energy_matrix()
    assert e.shape == (4, 16)
    assert _energy_row_orthonormality_error(e) < 1e-5


def test_orthogonal_default_optimizer_is_euclidean():
    model = Model(_toy_data(), latent_dim=3, energy_manifold="orthogonal")
    opt = model._default_optimizer(nu=1e-2)
    assert isinstance(opt, torch.optim.Adam)


@pytest.mark.parametrize(
    "name,cls",
    [
        ("adam", torch.optim.Adam),
        ("sgd", torch.optim.SGD),
    ],
)
def test_orthogonal_optimizer_string(name, cls):
    model = Model(_toy_data(), latent_dim=3, energy_manifold="orthogonal")
    opt = model._default_optimizer(1e-2, name)
    assert isinstance(opt, cls)


def test_orthogonal_fit_with_optimizer_sgd_string():
    model = Model(_toy_data(), latent_dim=4, energy_manifold="orthogonal")
    model.fit(its=80, nu=5e-2, optimizer="sgd", seed=0, gpu=False, track_loss=True)
    assert model.loss_history[-1] < model.loss_history[0]
    assert _energy_row_orthonormality_error(model.energy_matrix()) < 1e-4


def test_orthogonal_constraint_preserved_after_training():
    model = Model(_toy_data(), latent_dim=4, energy_manifold="orthogonal")
    model.fit(its=80, nu=5e-2, seed=0, gpu=False, track_loss=True)
    assert _energy_row_orthonormality_error(model.energy_matrix()) < 1e-4


def test_orthogonal_total_params_is_reduced():
    data = _toy_data(shape=(50, 10))
    m_plain = Model(data, latent_dim=3)
    m_orth = Model(data, latent_dim=3, energy_manifold="orthogonal")
    s, i = data.shape
    k = 3
    assert m_plain.total_params == s * k + k * i
    assert m_orth.total_params == s * k + (i * k - k * (k + 1) // 2)
    assert m_orth.total_params < m_plain.total_params


def test_orthogonal_requires_features_at_least_k():
    bad = np.zeros((50, 3), dtype=np.float32)  # features=3 < k=5
    with pytest.raises(ValueError, match="features >= latent_dim"):
        Model(bad, latent_dim=5, energy_manifold="orthogonal")


def test_plain_optimizer_passes_without_warning():
    model = Model(_toy_data(), latent_dim=3, energy_manifold="orthogonal")
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(its=5, optimizer=opt, gpu=False)


def test_draw_samples_works_on_orthogonal_model():
    model = Model(_toy_data(), latent_dim=3, energy_manifold="orthogonal").fit(
        its=20, nu=5e-2, seed=0, gpu=False
    )
    samples = model.draw_samples(n_samples=25, seed=1)
    assert samples.shape == (25, 16)
    assert set(np.unique(samples)).issubset({0, 1})
