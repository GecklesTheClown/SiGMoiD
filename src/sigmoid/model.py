"""
model.py

SiGMoiD Model Implementation

This module provides the Model class for fitting a SiGMoiD-based probabilistic
model to binary data. ``Model`` is a :class:`torch.nn.Module`, so it composes
with the wider PyTorch ecosystem (custom optimizers, LR schedulers, mixed
precision, ``nn.DataParallel``, etc.).

Example:
    import numpy as np
    from sigmoid import Model

    data = np.random.binomial(1, 0.5, size=(100, 50))

    # Default optimizer is Adam with lr=nu (nu defaults to 1e-2)
    model = Model(data, latent_dim=5).fit(its=1000, seed=42)

    # For higher throughput, bring your own SGD (faster per step but brittle to nu):
    import torch
    model = Model(data, latent_dim=5)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    model.fit(its=500, optimizer=opt, seed=42)

    aic_value = model.aic()
    samples = model.draw_samples(n_samples=10, seed=42)
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Model(nn.Module):
    """SiGMoiD model for binary data, implemented as an :class:`nn.Module`.

    Parameters
    ----------
    data : np.ndarray
        Binary data matrix of shape ``(samples, features)``. Values should be
        in ``{0, 1}``. Any numpy dtype is accepted and will be cast to float.
    latent_dim : int
        Dimensionality of the latent space (``k``).
    mean, std : float
        Mean and standard deviation used to initialize ``beta`` and ``energy``.

    Attributes
    ----------
    raw : torch.Tensor
        Float buffer holding the data (registered so ``.to(device)`` moves it).
    k : int
        Latent dimension.
    beta : nn.Parameter
        Sample-side latent matrix of shape ``(samples, k)``.
    energy : nn.Parameter
        Feature-side latent matrix of shape ``(k, features)``.
    loss_history : list[float]
        Per-iteration NLL recorded when ``fit(track_loss=True)``.
    """

    def __init__(
        self,
        data: np.ndarray,
        latent_dim: int,
        mean: float = 0.0,
        std: float = 0.01,
    ) -> None:
        super().__init__()
        self.register_buffer("raw", torch.from_numpy(np.asarray(data)).float())
        self.k = int(latent_dim)
        s, i = self.raw.shape
        self.beta = nn.Parameter(torch.empty(s, self.k).normal_(mean, std))
        self.energy = nn.Parameter(torch.empty(self.k, i).normal_(mean, std))
        self._init_mean = mean
        self._init_std = std
        self._fitted = False
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------ shape

    @property
    def total_params(self) -> int:
        """Total number of trainable parameters in the model."""
        s, i = self.raw.shape
        return s * self.k + self.k * i

    # ------------------------------------------------------------------ forward

    def forward(self) -> torch.Tensor:
        """Compute per-cell probabilities.

        Implements the SiGMoiD probability from the paper:

        .. math::
            p_{si} = \\frac{\\exp(-\\sum_k \\beta_{sk} E_{ki})}
                          {1 + \\exp(-\\sum_k \\beta_{sk} E_{ki})}

        which is equivalent to ``sigmoid(-(beta @ energy))``.
        """
        return torch.sigmoid(-(self.beta @ self.energy))

    # ------------------------------------------------------------------ derived state

    @property
    def prob_estimates(self) -> torch.Tensor | None:
        """Detached CPU copy of the current probability matrix, or ``None``."""
        if not self._fitted:
            return None
        with torch.no_grad():
            return self.forward().detach().cpu()

    @property
    def model_params(self) -> list[torch.Tensor] | None:
        """``[beta, energy]`` as detached CPU tensors, or ``None``."""
        if not self._fitted:
            return None
        return [self.beta.detach().cpu(), self.energy.detach().cpu()]

    # ------------------------------------------------------------------ training

    def _reinit_params(self, seed: int | None) -> None:
        """Re-initialize ``beta`` and ``energy`` in-place.

        In-place reinit preserves the underlying ``nn.Parameter`` object
        identities, so a user-supplied optimizer built before :meth:`fit` is
        called keeps tracking the right tensors. Assumes ``self.to(device)``
        has already been called.
        """
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            self.beta.normal_(self._init_mean, self._init_std)
            self.energy.normal_(self._init_mean, self._init_std)

    def fit(
        self,
        nu: float = 0.01,
        its: int = 2000,
        seed: int | None = None,
        gpu: bool = True,
        optimizer: torch.optim.Optimizer | None = None,
        track_loss: bool = False,
        verbose: bool = False,
    ) -> Model:
        """Fit the model by minimizing the summed binary cross-entropy.

        Equivalent to maximizing the log-likelihood of the binary data under
        the SiGMoiD probability ``p = sigmoid(-(beta @ energy))``.

        Args:
            nu: Learning rate used by the default :class:`~torch.optim.Adam`
                optimizer. Ignored if ``optimizer`` is supplied.
            its: Number of optimizer steps.
            seed: If given, seeds parameter re-initialization for reproducibility.
            gpu: Use CUDA if available.
            optimizer: Optional pre-built optimizer over ``self.parameters()``.
                If ``None``, defaults to ``torch.optim.Adam(self.parameters(), lr=nu)``.
                Adam is the default because it's robust to ``nu`` choice and
                tends to find a slightly lower NLL on typical SiGMoiD problems.
                On benchmarks (``examples/bench_optimizers.py``), well-tuned
                ``SGD(lr=nu)`` reaches the same plateau in roughly half the
                wall-time (no momentum/variance state to update each step) but
                is brittle to ``nu`` -- too high a value diverges. If you have
                a known-good ``nu`` and care about throughput, pass
                ``optimizer=torch.optim.SGD(model.parameters(), lr=nu)``
                explicitly. SGD with the same step size is also mathematically
                equivalent to the original hand-rolled SiGMoiD update.
            track_loss: If True, append the NLL of every iteration to
                ``self.loss_history`` (small per-iteration overhead).
            verbose: If True, log progress roughly every 10% of iterations.

        Returns:
            self, to allow chaining.
        """
        device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")
        self.to(device)
        self._reinit_params(seed=seed)

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=nu)
        else:
            # Sanity-check that the supplied optimizer is actually wired to our params.
            opt_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
            for p in self.parameters():
                if id(p) not in opt_param_ids:
                    raise ValueError(
                        "Supplied optimizer does not track all model parameters. "
                        "Build it after constructing the Model: "
                        "`opt = MyOpt(model.parameters(), ...)`."
                    )

        sigma = self.raw
        self.loss_history = []
        log_every = max(1, its // 10) if verbose else 0

        for it in range(its):
            optimizer.zero_grad()
            prob = self.forward()
            loss = F.binary_cross_entropy(prob, sigma, reduction="sum")
            loss.backward()
            optimizer.step()

            if track_loss:
                self.loss_history.append(loss.item())
            if log_every and (it + 1) % log_every == 0:
                msg = f"iter {it + 1}/{its}"
                if track_loss:
                    msg += f" nll={self.loss_history[-1]:.4f}"
                logger.info(msg)

        self._fitted = True
        return self

    # ------------------------------------------------------------------ scoring

    def log_likelihood(self) -> float:
        """Log-likelihood of the data under the fitted model."""
        if not self._fitted:
            raise ValueError("Model has not been fitted yet. Call fit first.")
        with torch.no_grad():
            prob = self.forward()
            ll = -F.binary_cross_entropy(prob, self.raw, reduction="sum")
        return float(ll)

    def aic(self) -> float:
        """Akaike Information Criterion: ``2 * total_params - 2 * log_likelihood``."""
        return float(2 * self.total_params - 2 * self.log_likelihood())

    # ------------------------------------------------------------------ sampling

    def draw_samples(self, n_samples: int = 1000, seed: int | None = None) -> np.ndarray:
        """Draw binary samples from the fitted model.

        Args:
            n_samples: Number of samples to draw.
            seed: Seed for the local NumPy RNG (does not touch global state).

        Returns:
            ``np.ndarray`` of shape ``(n_samples, features)`` with values in
            ``{0, 1}``.
        """
        if not self._fitted:
            raise ValueError("Model has not been fitted yet. Call fit first.")
        rng = np.random.default_rng(seed)
        s = self.beta.shape[0]
        indices = rng.integers(s, size=n_samples)
        with torch.no_grad():
            sample_betas = self.beta[indices]
            prob = torch.sigmoid(-(sample_betas @ self.energy)).detach().cpu().numpy()
        return rng.binomial(1, prob)
