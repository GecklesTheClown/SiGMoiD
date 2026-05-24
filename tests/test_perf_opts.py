"""Tests for optional bf16 autocast and torch.compile in Model.fit."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from sigmoid import Model


def _toy_data(seed: int = 0, shape=(50, 14)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.binomial(1, 0.4, size=shape).astype(np.float32)


def test_bf16_on_cpu_warns_and_trains():
    with pytest.warns(UserWarning, match="bf16=True has no effect on CPU"):
        model = Model(_toy_data(), latent_dim=3).fit(
            its=12, seed=0, gpu=False, bf16=True, track_loss=True
        )
    assert model.loss_history[-1] < model.loss_history[0]
    assert model.beta.dtype == torch.float32


def test_compile_model_on_cpu_runs():
    """Compile may fall back to eager when the inductor backend is unavailable."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = Model(_toy_data(), latent_dim=3).fit(
            its=12, seed=0, gpu=False, compile_model=True, track_loss=True
        )
    assert model.loss_history[-1] < model.loss_history[0]
    msgs = [str(w.message) for w in caught]
    assert any("torch.compile failed" in m for m in msgs) or not msgs


def test_stiefel_bf16_and_compile_on_cpu():
    pytest.importorskip("geoopt")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = Model(_toy_data(), latent_dim=3, energy_manifold="stiefel").fit(
            its=15,
            seed=0,
            gpu=False,
            bf16=True,
            compile_model=True,
            track_loss=True,
        )
    assert model.loss_history[-1] < model.loss_history[0]
    msgs = [str(w.message) for w in caught]
    assert any("bf16=True has no effect on CPU" in m for m in msgs)
    e = model.energy_matrix()
    k = e.shape[0]
    with torch.no_grad():
        err = (e @ e.T - torch.eye(k)).abs().max()
    assert float(err) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_cuda_smoke():
    model = Model(_toy_data(shape=(80, 20)), latent_dim=4).fit(
        its=20, seed=0, gpu=True, bf16=True, track_loss=True
    )
    assert model.loss_history[-1] < model.loss_history[0]
    assert model.beta.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stiefel_bf16_cuda_smoke():
    pytest.importorskip("geoopt")
    model = Model(_toy_data(shape=(80, 20)), latent_dim=4, energy_manifold="stiefel").fit(
        its=25, nu=5e-2, seed=0, gpu=True, bf16=True, track_loss=True
    )
    assert model.loss_history[-1] < model.loss_history[0]
    e = model.energy_matrix()
    k = e.shape[0]
    with torch.no_grad():
        err = (e @ e.T - torch.eye(k, device=e.device)).abs().max()
    assert float(err) < 1e-3
