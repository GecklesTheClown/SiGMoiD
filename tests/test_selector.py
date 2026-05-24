"""Smoke tests for the Selector interface."""

import numpy as np
import pytest

from sigmoid import Model, Selector


def _toy_data(seed: int = 0, shape=(40, 12)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.binomial(1, 0.4, size=shape).astype(np.float32)


def test_fit_runs_and_picks_optimal():
    data = _toy_data()
    sel = Selector(data, seed=42).fit(k=range(1, 4), its=20, repeats=2, gpu=False)
    assert isinstance(sel.optimal, Model)
    assert len(sel.trace) == 3 * 2


def test_trace_records_lat_dim_aic_seed():
    data = _toy_data()
    sel = Selector(data, seed=1).fit(k=[1, 2], its=10, repeats=1, gpu=False)
    for latent_dim, aic, seed in sel.trace:
        assert latent_dim in (1, 2)
        assert isinstance(aic, float)
        assert isinstance(seed, int)


def test_fit_with_bic_criterion():
    data = _toy_data()
    sel = Selector(data, seed=1).fit(
        k=[1, 2], its=10, repeats=1, gpu=False, criterion="bic"
    )
    assert sel.criterion == "bic"
    for latent_dim, score, seed in sel.trace:
        assert latent_dim in (1, 2)
        assert isinstance(score, float)
        assert isinstance(seed, int)


def test_invalid_criterion_raises():
    with pytest.raises(ValueError, match="criterion must be"):
        Selector(_toy_data()).fit(k=[1], its=5, repeats=1, gpu=False, criterion="cv")


def test_keep_all_retains_candidates():
    data = _toy_data()
    sel = Selector(data, seed=0).fit(
        k=[1, 2], its=10, repeats=2, gpu=False, keep_all=True
    )
    assert len(sel.candidates) == 4
    sel2 = Selector(data, seed=0).fit(k=[1, 2], its=10, repeats=2, gpu=False)
    assert sel2.candidates == {}


def test_seed_reproducibility():
    data = _toy_data()
    a = Selector(data, seed=99).fit(k=[1, 2], its=10, repeats=2, gpu=False)
    b = Selector(data, seed=99).fit(k=[1, 2], its=10, repeats=2, gpu=False)
    assert [t[:2] for t in a.trace] == [t[:2] for t in b.trace]


def test_gpu_false_is_honored(monkeypatch):
    """Regression: Selector.fit must forward gpu=False to Model.fit."""
    seen = {}
    original = Model.fit

    def spy(self, *args, **kwargs):
        seen["gpu"] = kwargs.get("gpu", True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Model, "fit", spy)
    Selector(_toy_data(), seed=0).fit(k=[1], its=5, repeats=1, gpu=False)
    assert seen["gpu"] is False
