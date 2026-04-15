"""Compare SGD vs Adam convergence on a synthetic SiGMoiD dataset.

Generates a low-rank-structured binary dataset (true latent dim k_true), then
fits Models with several optimizer/lr combinations across multiple seeds and
reports loss trajectories, final NLL, iterations-to-target, and wall time.

Run:
    uv run python examples/bench_optimizers.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from sigmoid import Model

# ---------------------------------------------------------------------- config

N_SAMPLES, N_FEATURES = 200, 60
K_TRUE = 4              # data-generating latent dim
K_FIT = 4               # latent dim used by the fitted Model
ITS = 1500
SEEDS = [0, 1, 2, 3, 4]
DEVICE_GPU = False      # benchmarking on CPU for reproducibility


@dataclass
class Run:
    name: str
    final_nll: float
    nll_at_quartiles: tuple[float, float, float]   # NLL at 25%, 50%, 75% of iters
    iters_to_99pct: int | None                     # iters to reach 99% of total reduction
    wall_s: float


def make_data(seed: int) -> np.ndarray:
    """Generate binary data with genuine low-rank structure."""
    rng = np.random.default_rng(seed)
    beta_t = rng.normal(0, 1.0, size=(N_SAMPLES, K_TRUE)).astype(np.float32)
    energy_t = rng.normal(0, 1.0, size=(K_TRUE, N_FEATURES)).astype(np.float32)
    z = beta_t @ energy_t                       # logits
    p = 1.0 / (1.0 + np.exp(z))                 # paper form: sigmoid(-z)
    return rng.binomial(1, p).astype(np.float32)


def train(data: np.ndarray, seed: int, opt_factory, name: str) -> Run:
    """Train one Model with the given optimizer factory; return summary stats."""
    model = Model(data, latent_dim=K_FIT)
    opt = opt_factory(model.parameters())

    # We need parity in *initialization* across optimizers, so seed and
    # reinit explicitly via fit(seed=...).
    t0 = time.perf_counter()
    model.fit(its=ITS, optimizer=opt, seed=seed, gpu=DEVICE_GPU, track_loss=True)
    wall = time.perf_counter() - t0

    history = model.loss_history
    nll0 = history[0]
    final = history[-1]
    total_drop = nll0 - final

    iters_to_99 = None
    if total_drop > 0:
        target = nll0 - 0.99 * total_drop
        for i, v in enumerate(history):
            if v <= target:
                iters_to_99 = i + 1
                break

    q1, q2, q3 = ITS // 4, ITS // 2, 3 * ITS // 4
    return Run(
        name=name,
        final_nll=final,
        nll_at_quartiles=(history[q1], history[q2], history[q3]),
        iters_to_99pct=iters_to_99,
        wall_s=wall,
    )


def summarize(runs: list[Run]) -> dict[str, float]:
    finals = np.array([r.final_nll for r in runs])
    walls = np.array([r.wall_s for r in runs])
    iters = [r.iters_to_99pct for r in runs if r.iters_to_99pct is not None]
    return {
        "final_nll_mean": float(finals.mean()),
        "final_nll_std": float(finals.std()),
        "wall_s_mean": float(walls.mean()),
        "iters_to_99_median": float(np.median(iters)) if iters else float("nan"),
    }


def main() -> None:
    torch.set_num_threads(1)  # reduce noise from BLAS thread scheduling

    optimizers = {
        "SGD lr=1e-3 (paper default nu)": lambda p: torch.optim.SGD(p, lr=1e-3),
        "SGD lr=1e-2":                    lambda p: torch.optim.SGD(p, lr=1e-2),
        "SGD lr=1e-1":                    lambda p: torch.optim.SGD(p, lr=1e-1),
        "Adam lr=1e-3":                   lambda p: torch.optim.Adam(p, lr=1e-3),
        "Adam lr=1e-2":                   lambda p: torch.optim.Adam(p, lr=1e-2),
        "Adam lr=1e-1":                   lambda p: torch.optim.Adam(p, lr=1e-1),
        "AdamW lr=1e-2":                  lambda p: torch.optim.AdamW(p, lr=1e-2),
    }

    results: dict[str, list[Run]] = {name: [] for name in optimizers}
    histories: dict[str, list[list[float]]] = {name: [] for name in optimizers}

    for seed in SEEDS:
        data = make_data(seed=seed + 100)  # data seed independent of init seed
        for name, factory in optimizers.items():
            run = train(data, seed=seed, opt_factory=factory, name=name)
            results[name].append(run)
            # Re-train once to capture full history for plotting (cheap enough)
            m = Model(data, latent_dim=K_FIT)
            opt = factory(m.parameters())
            m.fit(its=ITS, optimizer=opt, seed=seed, gpu=DEVICE_GPU, track_loss=True)
            histories[name].append(m.loss_history)

    # --- summary table ----------------------------------------------------
    print(f"\nDataset: ({N_SAMPLES}x{N_FEATURES}), k_true={K_TRUE}, k_fit={K_FIT}, "
          f"its={ITS}, seeds={len(SEEDS)}\n")
    header = f"{'optimizer':<32} {'final NLL':>14} {'iters_to_99%':>14} {'wall (s)':>10}"
    print(header)
    print("-" * len(header))
    for name, runs in results.items():
        s = summarize(runs)
        iters_med = s["iters_to_99_median"]
        iters_str = f"{iters_med:.0f}" if not np.isnan(iters_med) else "n/a"
        print(
            f"{name:<32} "
            f"{s['final_nll_mean']:>9.2f}+/-{s['final_nll_std']:<4.2f} "
            f"{iters_str:>14} "
            f"{s['wall_s_mean']:>10.3f}"
        )

    # --- save loss curves -------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5.5))
        # Clip y so divergent runs don't squash the rest.
        all_finals = [h[-1] for hs in histories.values() for h in hs]
        ymax = float(np.percentile(all_finals, 75)) * 3
        for name, hs in histories.items():
            mean_curve = np.mean(hs, axis=0)
            ax.plot(mean_curve, label=name, linewidth=1.4)
        ax.set_ylim(top=ymax)
        ax.set_xlabel("iteration")
        ax.set_ylabel("NLL (sum BCE)")
        ax.set_title(f"Optimizer convergence on {N_SAMPLES}x{N_FEATURES} binary data, k={K_FIT}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        out = "examples/bench_optimizers.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"\nLoss curves written to {out}")

        # log-scale view of the gap above the best final NLL
        best_final = min(min(h) for hs in histories.values() for h in hs)
        fig2, ax2 = plt.subplots(figsize=(9, 5.5))
        for name, hs in histories.items():
            mean_curve = np.mean(hs, axis=0)
            gap = mean_curve - best_final + 1e-3
            ax2.semilogy(gap, label=name, linewidth=1.4)
        ax2.set_xlabel("iteration")
        ax2.set_ylabel("NLL - best_final + 1e-3 (log)")
        ax2.set_title("Convergence gap above best run (log scale)")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(alpha=0.3, which="both")
        fig2.tight_layout()
        out2 = "examples/bench_optimizers_loggap.png"
        fig2.savefig(out2, dpi=120)
        print(f"Log-gap curves written to {out2}")
    except ImportError:
        print("\n(matplotlib not installed — skipping plots)")


if __name__ == "__main__":
    main()
