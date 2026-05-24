"""Tests for Stiefel-manifold-constrained energy.

Skipped entirely if geoopt is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sigmoid import Model

geoopt = pytest.importorskip("geoopt")


def _toy_data(seed: int = 0, shape=(60, 16)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.binomial(1, 0.4, size=shape).astype(np.float32)


def _energy_row_orthonormality_error(energy_ki: torch.Tensor) -> float:
    """Max abs entry of ``E @ E^T - I`` for ``E`` of shape ``(k, features)``."""
    with torch.no_grad():
        k = energy_ki.shape[0]
        err = energy_ki @ energy_ki.T - torch.eye(k, device=energy_ki.device, dtype=energy_ki.dtype)
    return float(err.abs().max())


def test_stiefel_energy_T_init_is_on_manifold():
    model = Model(_toy_data(), latent_dim=4, energy_manifold="stiefel")
    assert isinstance(model.energy_T, geoopt.ManifoldParameter)
    assert model.energy_T.shape == (16, 4)
    e = model.energy_matrix()
    assert e.shape == (4, 16)
    assert _energy_row_orthonormality_error(e) < 1e-5


def test_stiefel_default_optimizer_is_riemannian():
    model = Model(_toy_data(), latent_dim=3, energy_manifold="stiefel")
    opt = model._default_optimizer(nu=1e-2)
    assert isinstance(opt, geoopt.optim.RiemannianAdam)


@pytest.mark.parametrize(
    "name,cls",
    [
        ("adam", geoopt.optim.RiemannianAdam),
        ("sgd", geoopt.optim.RiemannianSGD),
    ],
)
def test_stiefel_optimizer_string(name, cls):
    model = Model(_toy_data(), latent_dim=3, energy_manifold="stiefel")
    opt = model._default_optimizer(1e-2, name)
    assert isinstance(opt, cls)


def test_stiefel_fit_with_optimizer_sgd_string():
    model = Model(_toy_data(), latent_dim=4, energy_manifold="stiefel")
    model.fit(its=80, nu=5e-2, optimizer="sgd", seed=0, gpu=False, track_loss=True)
    assert model.loss_history[-1] < model.loss_history[0]
    assert _energy_row_orthonormality_error(model.energy_matrix()) < 1e-4


def test_stiefel_constraint_preserved_after_training():
    model = Model(_toy_data(), latent_dim=4, energy_manifold="stiefel")
    model.fit(its=80, nu=5e-2, seed=0, gpu=False, track_loss=True)
    assert _energy_row_orthonormality_error(model.energy_matrix()) < 1e-4


def test_stiefel_training_reduces_nll():
    model = Model(_toy_data(shape=(120, 30)), latent_dim=4, energy_manifold="stiefel")
    model.fit(its=150, nu=5e-2, seed=0, gpu=False, track_loss=True)
    assert model.loss_history[-1] < model.loss_history[0]


def test_stiefel_total_params_is_reduced():
    """AIC should use reduced energy DOF: ki - k(k+1)/2 instead of ki."""
    data = _toy_data(shape=(50, 10))
    m_plain = Model(data, latent_dim=3)
    m_stiefel = Model(data, latent_dim=3, energy_manifold="stiefel")
    s, i = data.shape
    k = 3
    assert m_plain.total_params == s * k + k * i
    assert m_stiefel.total_params == s * k + (i * k - k * (k + 1) // 2)
    assert m_stiefel.total_params < m_plain.total_params


def test_stiefel_requires_features_at_least_k():
    bad = np.zeros((50, 3), dtype=np.float32)  # features=3 < k=5
    with pytest.raises(ValueError, match="features >= latent_dim"):
        Model(bad, latent_dim=5, energy_manifold="stiefel")


def test_beta_manifold_removed():
    with pytest.raises(ValueError, match="beta_manifold was removed"):
        Model(_toy_data(), latent_dim=2, beta_manifold="stiefel")


def test_unknown_manifold_rejected():
    with pytest.raises(ValueError, match="Unknown energy_manifold"):
        Model(_toy_data(), latent_dim=2, energy_manifold="hyperbolic")


def test_warn_when_euclidean_optimizer_used_with_manifold():
    model = Model(_toy_data(), latent_dim=3, energy_manifold="stiefel")
    bad_opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    with pytest.warns(RuntimeWarning, match="not a geoopt Riemannian optimizer"):
        model.fit(its=5, optimizer=bad_opt, gpu=False)


def test_riemannian_optimizer_passes_without_warning():
    import warnings

    model = Model(_toy_data(), latent_dim=3, energy_manifold="stiefel")
    opt = geoopt.optim.RiemannianAdam(model.parameters(), lr=1e-2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(its=5, optimizer=opt, gpu=False)


def test_plain_model_unaffected_when_geoopt_installed():
    model = Model(_toy_data(), latent_dim=3).fit(its=10, seed=0, gpu=False)
    assert model.prob_estimates is not None


def test_draw_samples_works_on_manifold_model():
    model = Model(_toy_data(), latent_dim=3, energy_manifold="stiefel").fit(
        its=20, nu=5e-2, seed=0, gpu=False
    )
    samples = model.draw_samples(n_samples=25, seed=1)
    assert samples.shape == (25, 16)
    assert set(np.unique(samples)).issubset({0, 1})
