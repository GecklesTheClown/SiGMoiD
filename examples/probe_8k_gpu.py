"""Quick GPU timing probe for large SiGMoiD fits (bf16 / compile)."""

from __future__ import annotations

import time

import numpy as np
import torch

from sigmoid import Model

S, I, K = 8000, 8000, 30
ITS_PROBE = 100
ITS_EXTRAP = 2000
N_CANDIDATES = 60 * 10  # k grid x repeats


def main() -> None:
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))
        print("bf16 supported:", torch.cuda.is_bf16_supported())

    rng = np.random.default_rng(0)
    print(f"Allocating data ({S}x{I})...")
    data = rng.binomial(1, 0.5, size=(S, I)).astype(np.float32)
    print(f"Data size: {data.nbytes / 1e6:.1f} MB")

    configs = [
        ("baseline", False, False),
        ("bf16", True, False),
        ("bf16+compile", True, True),
    ]

    for label, bf16, compile_model in configs:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model = Model(data, latent_dim=K).fit(
            its=ITS_PROBE,
            seed=0,
            gpu=True,
            bf16=bf16,
            compile_model=compile_model,
            track_loss=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        nll0, nll1 = model.loss_history[0], model.loss_history[-1]
        per_iter = elapsed / ITS_PROBE
        est_2k = per_iter * ITS_EXTRAP
        est_grid = est_2k * N_CANDIDATES
        print(f"--- {label} ---")
        print(f"  {ITS_PROBE} iters: {elapsed:.2f}s ({per_iter * 1000:.1f} ms/iter)")
        print(f"  extrap 2000 iters / fit: {est_2k / 60:.1f} min")
        print(f"  extrap {N_CANDIDATES} fits: {est_grid / 3600:.1f} hours")
        print(f"  NLL {nll0:.2e} -> {nll1:.2e}")


if __name__ == "__main__":
    main()
