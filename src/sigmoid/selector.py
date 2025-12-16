"""
selector.py

SiGMoiD model selection module, for selecting the optimal latent dimension based on AIC.

Example:
    import numpy as np
    from sigmoid.selector import Selector

    # Sample binary data
    data = np.random.binomial(1, 0.5, size=(100, 50))

    # Initialize and fit the selector
    selector = Selector(data, seed=42)
    selector.fit(k=range(1, 11), its=500, repeats=5, verbose=True)

    # Access the optimal model
    optimal_model = selector.optimal
    print(f"Optimal latent dimension: {optimal_model.k}")
"""

import numpy as np
import torch
from . import model


class Selector:
    """
    SiGMoiD Model Selector for choosing optimal latent dimension.

    Attributes
    ----------
    raw : np.ndarray
        Binary data matrix.
    index : np.ndarray
        Shape of the data matrix.
    optimal : model.Model
        The optimal SiGMoiD model after fitting.
    trace : list
        List of tuples recording (latent_dim, AIC, seed) for each candidate model.
    master_seed : int
        Seed for random number generation to ensure reproducibility.
    """

    def __init__(self, data, seed=None):
        """Initialize the Selector with data and an optional random seed.

        Args:
            data (np.ndarray): Binary data matrix of shape (samples, features).
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        self.raw = data
        self.index = np.array([data.shape[0], data.shape[1]])
        self.optimal = None
        self.trace = []
        self.master_seed = seed

    def fit(self, k=range(1, 21), its=2000, repeats=10, gpu=True, verbose=False):
        """_summary_

        Args:
            k (iterable, optional): Range or list of latent dimensions to evaluate. Defaults to range(1, 21).
            its (int, optional): Number of iterations for training each model. Defaults to 2000.
            repeats (int, optional): Number of repeats for each latent dimension to ensure robustness. Defaults to 10.
            gpu (bool, optional): Whether to use GPU for training if available. Defaults to True.
            verbose (bool, optional): Whether to print progress messages. Defaults to False.
        """

        # RNG and array setup
        k_range = np.asarray(k)
        master_rng = np.random.default_rng(seed=self.master_seed)
        init_seeds = master_rng.integers(2**32 - 1, size=(len(k), repeats))
        leading_candidate_aic = None

        for i, latent_dim in enumerate(k_range):
            for j in range(repeats):

                if verbose:
                    print(
                        f"Training candidate model with latent dim {latent_dim}, "
                        f"repeat {j+1}/{repeats} (seed {init_seeds[i,j]})"
                    )

                candidate = model.Model(self.raw, latent_dim=latent_dim)
                candidate.train(k=latent_dim, seed=init_seeds[i, j], its=its)
                candidate_aic = candidate.aic()
                self.trace.append((latent_dim, candidate_aic, init_seeds[i, j]))

                if (
                    leading_candidate_aic is None
                    or candidate_aic < leading_candidate_aic
                ):
                    leading_candidate_aic = candidate_aic
                    self.optimal = candidate
