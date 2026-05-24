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

    # Constrain beta to the Stiefel manifold (orthonormal columns).
    # Requires the optional `geoopt` dependency.
    model = Model(data, latent_dim=5, beta_manifold="stiefel").fit(its=500, seed=42)
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Optional geoopt support for manifold-constrained parameters. We import
# defensively so the package works without geoopt installed.
try:
    import geoopt
    from geoopt.optim.mixin import OptimMixin as _GeooptOptimMixin

    _HAS_GEOOPT = True
except ImportError:  # pragma: no cover - covered by tests that skip
    geoopt = None  # type: ignore[assignment]
    _GeooptOptimMixin = None  # type: ignore[assignment]
    _HAS_GEOOPT = False


_SUPPORTED_MANIFOLDS = ("stiefel",)


def _require_geoopt(feature: str) -> None:
    if not _HAS_GEOOPT:
        raise ImportError(
            f"{feature} requires the optional `geoopt` dependency. "
            "Install with `pip install sigmoid-py[geometry]` or `pip install geoopt`."
        )


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
        Mean and standard deviation used to initialize ``beta`` and ``energy``
        when no manifold constraint is applied.
    beta_manifold : {None, "stiefel"}, optional
        If ``"stiefel"``, constrain ``beta`` to the Stiefel manifold
        :math:`\\mathrm{St}(s, k) = \\{ B \\in \\mathbb{R}^{s \\times k} :
        B^\\top B = I_k \\}`. Requires ``s >= k`` and the optional ``geoopt``
        dependency. Removes rotational ambiguity in the latent space without
        losing expressivity (any unconstrained factorization ``beta @ energy``
        can be rewritten as ``Q @ (R @ energy)`` via QR decomposition, with
        ``Q`` on the Stiefel manifold).

    Attributes
    ----------
    raw : torch.Tensor
        Float buffer holding the data (registered so ``.to(device)`` moves it).
    k : int
        Latent dimension.
    beta : nn.Parameter or geoopt.ManifoldParameter
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
        beta_manifold: str | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("raw", torch.from_numpy(np.asarray(data)).float())
        self.k = int(latent_dim)
        s, i = self.raw.shape

        if beta_manifold is not None and beta_manifold not in _SUPPORTED_MANIFOLDS:
            raise ValueError(
                f"Unknown beta_manifold={beta_manifold!r}. "
                f"Supported: {_SUPPORTED_MANIFOLDS} or None."
            )
        self.beta_manifold = beta_manifold

        if beta_manifold == "stiefel":
            _require_geoopt("beta_manifold='stiefel'")
            if s < self.k:
                raise ValueError(
                    f"Stiefel manifold for beta requires samples >= latent_dim "
                    f"(got samples={s}, latent_dim={self.k})."
                )
            stiefel = geoopt.Stiefel()
            self.beta = geoopt.ManifoldParameter(
                stiefel.random(s, self.k), manifold=stiefel
            )
        else:
            self.beta = nn.Parameter(torch.empty(s, self.k).normal_(mean, std))

        # Energy is always Euclidean. Putting it on Stiefel too would overconstrain
        # the span of `beta @ energy`; leaving it free absorbs any scale/rotation
        # that the Stiefel gauge-fix on beta pushes out.
        self.energy = nn.Parameter(torch.empty(self.k, i).normal_(mean, std))

        self._init_mean = mean
        self._init_std = std
        self._fitted = False
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------ shape

    @property
    def total_params(self) -> int:
        """Effective number of free parameters (used in AIC and BIC).

        For an unconstrained ``beta`` this is ``s*k + k*i``. When ``beta`` is
        constrained to the Stiefel manifold, its effective dimension drops to
        ``s*k - k*(k+1)/2`` (see Edelman et al. 1998), and the AIC adjusts
        accordingly so that model selection is fair between constrained and
        unconstrained fits.
        """
        s, i = self.raw.shape
        if self.beta_manifold == "stiefel":
            beta_dof = s * self.k - self.k * (self.k + 1) // 2
        else:
            beta_dof = s * self.k
        return beta_dof + self.k * i

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
            if self.beta_manifold == "stiefel":
                # Sample a fresh point on the Stiefel manifold.
                assert _HAS_GEOOPT  # guaranteed by __init__
                new_beta = self.beta.manifold.random(*self.beta.shape).to(self.beta.device)
                self.beta.copy_(new_beta)
            else:
                self.beta.normal_(self._init_mean, self._init_std)
            self.energy.normal_(self._init_mean, self._init_std)

    def _default_optimizer(self, nu: float) -> torch.optim.Optimizer:
        """Pick an appropriate default optimizer for our parameters."""
        if self.beta_manifold is not None:
            _require_geoopt("default optimizer for manifold-constrained beta")
            # RiemannianAdam handles mixed Euclidean + manifold params: it
            # applies the manifold's retraction to ManifoldParameters and
            # falls back to a standard Adam step for plain nn.Parameters.
            return geoopt.optim.RiemannianAdam(self.parameters(), lr=nu)
        return torch.optim.Adam(self.parameters(), lr=nu)

    def _validate_user_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Check that a user-supplied optimizer is compatible with our params."""
        opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        for p in self.parameters():
            if id(p) not in opt_param_ids:
                raise ValueError(
                    "Supplied optimizer does not track all model parameters. "
                    "Build it after constructing the Model: "
                    "`opt = MyOpt(model.parameters(), ...)`."
                )
        # If beta lives on a manifold, a non-Riemannian optimizer will silently
        # drift off the manifold. Warn loudly.
        if (
            self.beta_manifold is not None
            and _HAS_GEOOPT
            and not isinstance(optimizer, _GeooptOptimMixin)
        ):
            warnings.warn(
                "Model was constructed with beta_manifold="
                f"{self.beta_manifold!r} but the supplied optimizer "
                f"({type(optimizer).__name__}) is not a geoopt Riemannian "
                "optimizer. The manifold constraint on `beta` will drift "
                "during training. Use geoopt.optim.RiemannianAdam or "
                "geoopt.optim.RiemannianSGD instead.",
                RuntimeWarning,
                stacklevel=3,
            )

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
            nu: Learning rate used by the default optimizer. Ignored if
                ``optimizer`` is supplied.
            its: Number of optimizer steps.
            seed: If given, seeds parameter re-initialization for reproducibility.
            gpu: Use CUDA if available.
            optimizer: Optional pre-built optimizer over ``self.parameters()``.
                Defaults depend on whether ``beta`` is manifold-constrained:

                * Unconstrained: :class:`torch.optim.Adam` (``lr=nu``). Robust
                  to ``nu`` and slightly better final NLL than SGD on typical
                  problems. Plain ``SGD(lr=nu)`` is ~2x faster per iteration
                  and mathematically equivalent to the original hand-rolled
                  SiGMoiD update, if you have a known-good ``nu``.
                * Stiefel-constrained ``beta``:
                  :class:`geoopt.optim.RiemannianAdam` (``lr=nu``), which
                  applies the Stiefel retraction after each Adam step so the
                  constraint is preserved. ``RiemannianSGD`` is the other
                  geoopt-provided option. ``AdamW`` is not appropriate for
                  manifold-constrained parameters because decoupled weight
                  decay shrinks towards zero, leaving the manifold.
            track_loss: If True, append the NLL of every iteration to
                ``self.loss_history``.
            verbose: If True, log progress roughly every 10% of iterations.

        Returns:
            self, to allow chaining.
        """
        device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")
        self.to(device)
        self._reinit_params(seed=seed)

        if optimizer is None:
            optimizer = self._default_optimizer(nu)
        else:
            self._validate_user_optimizer(optimizer)

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
        """Akaike Information Criterion: ``2 * total_params - 2 * log_likelihood``.

        When ``beta`` is manifold-constrained, ``total_params`` already
        accounts for the reduced degrees of freedom (see :attr:`total_params`).
        """
        return float(2 * self.total_params - 2 * self.log_likelihood())

    def bic(self) -> float:
        """Bayesian Information Criterion: ``ln(n) * total_params - 2 * log_likelihood``.

        Here ``n`` is the number of samples (rows in the data matrix). Uses the
        same :attr:`total_params` as :meth:`aic`, including Stiefel reductions.
        """
        n = int(self.raw.shape[0])
        return float(np.log(n) * self.total_params - 2 * self.log_likelihood())

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
