"""Smoke tests for the Model interface."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from sigmoid import Model


def _toy_data(seed: int = 0, shape=(40, 12)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.binomial(1, 0.4, size=shape).astype(np.float32)


def test_model_is_nn_module_with_parameters():
    model = Model(_toy_data(), latent_dim=3)
    assert isinstance(model, nn.Module)
    params = list(model.parameters())
    assert len(params) == 2
    assert all(isinstance(p, nn.Parameter) and p.requires_grad for p in params)


def test_fit_returns_self_and_populates_state():
    data = _toy_data()
    model = Model(data, latent_dim=3).fit(its=20, seed=0, gpu=False)
    assert isinstance(model, Model)
    assert model.prob_estimates is not None
    assert model.model_params is not None
    assert model.prob_estimates.shape == data.shape


def test_fit_is_deterministic_with_seed():
    data = _toy_data()
    a = Model(data, latent_dim=2).fit(its=30, seed=123, gpu=False)
    b = Model(data, latent_dim=2).fit(its=30, seed=123, gpu=False)
    assert torch.allclose(a.prob_estimates, b.prob_estimates)


def test_int_input_is_accepted():
    """Regression: integer dtype input must not break BCE."""
    rng = np.random.default_rng(0)
    data_int = rng.binomial(1, 0.5, size=(20, 8))  # int64
    model = Model(data_int, latent_dim=2).fit(its=10, seed=0, gpu=False)
    assert isinstance(model.aic(), float)


def test_bic_after_fit():
    data = _toy_data()
    model = Model(data, latent_dim=2).fit(its=20, seed=0, gpu=False)
    n = data.shape[0]
    expected = np.log(n) * model.total_params - 2 * model.log_likelihood()
    assert model.bic() == pytest.approx(expected)


def test_log_likelihood_before_fit_raises():
    model = Model(_toy_data(), latent_dim=2)
    with pytest.raises(ValueError):
        model.log_likelihood()


def test_draw_samples_shape_and_values():
    data = _toy_data()
    model = Model(data, latent_dim=2).fit(its=20, seed=0, gpu=False)
    samples = model.draw_samples(n_samples=50, seed=1)
    assert samples.shape == (50, data.shape[1])
    assert set(np.unique(samples)).issubset({0, 1})


def test_draw_samples_before_fit_raises():
    with pytest.raises(ValueError):
        Model(_toy_data(), latent_dim=2).draw_samples(n_samples=5)


def test_draw_samples_does_not_mutate_global_numpy_rng():
    model = Model(_toy_data(), latent_dim=2).fit(its=10, seed=0, gpu=False)
    np.random.seed(7)
    before = np.random.rand()
    np.random.seed(7)
    _ = model.draw_samples(n_samples=10, seed=42)
    after = np.random.rand()
    assert before == after


def test_loss_history_tracking_optional():
    data = _toy_data()
    m = Model(data, latent_dim=2).fit(its=15, seed=0, gpu=False, track_loss=True)
    assert len(m.loss_history) == 15
    m_off = Model(data, latent_dim=2).fit(its=15, seed=0, gpu=False)
    assert m_off.loss_history == []


def test_training_actually_minimizes_nll():
    """Sanity check: fit must lower NLL on a learnable dataset.

    Guards against an inverted forward pass (the paper's probability is
    ``sigmoid(-(beta @ energy))``, not ``sigmoid(beta @ energy)``) and against
    optimizer/loss sign mistakes.
    """
    data = _toy_data(shape=(80, 24))
    m = Model(data, latent_dim=4).fit(its=300, nu=1e-2, seed=0, gpu=False, track_loss=True)
    assert m.loss_history[-1] < m.loss_history[0]
    # Final NLL should beat the chance-level (uniform 0.5) baseline.
    chance = -float(
        torch.nn.functional.binary_cross_entropy(
            torch.full_like(torch.tensor(data), 0.5),
            torch.tensor(data),
            reduction="sum",
        )
    )
    assert m.log_likelihood() > chance


def test_custom_optimizer_is_accepted():
    data = _toy_data()
    model = Model(data, latent_dim=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.fit(its=20, optimizer=opt, gpu=False, track_loss=True)
    assert model.loss_history[-1] < model.loss_history[0]


def test_mismatched_optimizer_is_rejected():
    """Supplying an optimizer that doesn't track our params must error early."""
    model = Model(_toy_data(), latent_dim=2)
    stray = nn.Parameter(torch.zeros(3))
    bad_opt = torch.optim.SGD([stray], lr=0.01)
    with pytest.raises(ValueError, match="does not track"):
        model.fit(its=5, optimizer=bad_opt, gpu=False)


def test_state_dict_roundtrips():
    """Standard nn.Module persistence should just work."""
    data = _toy_data()
    a = Model(data, latent_dim=2).fit(its=10, seed=0, gpu=False)
    b = Model(data, latent_dim=2)
    b.load_state_dict(a.state_dict())
    b._fitted = True  # mark as fitted since we loaded trained params
    assert torch.allclose(a.prob_estimates, b.prob_estimates)
