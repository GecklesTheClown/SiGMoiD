"""
model.py

SiGMoiD Model Implementation

This module provides the Model class for fitting a SiGMoiD-based probabilistic model to binary data.

Example:
    import numpy as np
    from sigmoid.model import Model

    # Sample binary data
    data = np.random.binomial(1, 0.5, size=(100, 50))

    # Initialize and train the model
    model = Model(data, latent_dim=5)
    model.train(its=1000, seed=42)

    # Compute AIC
    aic_value = model.aic()
    print(f"AIC: {aic_value}")

    # Draw samples from the trained model
    samples = model.draw_samples(n_samples=10, seed=42)
    print(samples)
"""

import numpy as np
import torch
import torch.nn.functional as F


class Model:
    """
    SiGMoiD Model for binary data.

    Attributes
    ----------
    raw : torch.Tensor
        Binary data matrix as a torch tensor.
    index : np.ndarray
        Shape of the data matrix.
    k : int
        Dimensionality of the latent space.
    total_params : int
        Total number of parameters in the model.
    prob_estimates : torch.Tensor
        Estimated probabilities after training.
    model_params : list
        List containing model parameters [beta, energy].
    """

    def __init__(self, data, latent_dim):
        """
        Initialize the SiGMoiD model with binary data and latent dimension.

        Args:
            data (np.ndarray): Binary data matrix of shape (samples, features).
            latent_dim (int): Dimensionality of the latent space.
        """
        self.raw = torch.from_numpy(data)
        self.index = np.array([data.shape[0], data.shape[1]])
        self.k = latent_dim
        self.total_params = self.param_counter()
        self.prob_estimates = None
        self.model_params = None

    def param_counter(self):
        """Calculate total number of parameters in the model."""
        s = self.raw.shape[0]
        i = self.raw.shape[1]
        return (s * self.k) + (self.k * i)

    def aic(self):
        """Compute Akaike Information Criterion for the model."""
        ll = self.log_likelihood()
        aic = 2 * self.total_params - 2 * ll
        return aic

    def log_likelihood(self):
        """ "Compute the log-likelihood of the observed data given the model."""
        if self.prob_estimates is None:
            raise ValueError("Model has not been fitted yet. Call train first.")

        probs = self.prob_estimates
        sigma = self.raw.float()

        # Compute log-likelihood using binary cross-entropy for numerical stability
        ll = -F.binary_cross_entropy(probs, sigma, reduction="sum")
        return ll

    def train(self, nu=0.001, its=2000, k=1, seed=None, gpu=True, mean=0.0, std=0.01):
        """Search for model parameters using gradient ascent using the log-liklelihood as the objective function.

        Args:
            nu (float, optional): Step size. Defaults to 0.001.
            its (int, optional): Number of iterations. Defaults to 2000.
            k (int, optional): Latent dimension. Defaults to 1.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            gpu (bool, optional): Whether to use GPU if available. Defaults to True.
            mean (float, optional): Mean for parameter initialization. Defaults to 0.0.
            std (float, optional): Standard deviation for parameter initialization. Defaults to 0.01.
        """
        s = self.index[0]
        i = self.index[1]

        if seed is not None:
            torch.manual_seed(seed)

        if gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        sigma = self.raw.to(device)
        if self.prob_estimates is not None:
            self.prob_estimates = self.prob_estimates.to(device)

        beta = torch.normal(mean=mean, std=std, size=(s, k), device=device)
        energy = torch.normal(mean=mean, std=std, size=(k, i), device=device)
        prob_sigma = self._sigma_probability(beta, energy)

        for _ in range(its):
            diff = prob_sigma - sigma
            beta_init = beta.clone()
            # diff @ energy: gradient matrix w.r.t. beta
            beta = beta + nu * (diff @ energy.T)
            # beta_init.T @ diff: gradient matrix w.r.t. energy
            energy = energy + nu * (beta_init.T @ diff)
            prob_sigma = self._sigma_probability(beta, energy)

        self.prob_estimates = prob_sigma.cpu().detach()
        self.model_params = [beta.cpu().detach(), energy.cpu().detach()]

    def _sigmoid_transform(self, x):
        """Numerically stable sigmoid transformation."""
        return torch.where(
            x >= 0, torch.exp(-x) / (1 + torch.exp(-x)), 1 / (1 + torch.exp(x))
        )

    def _sigma_probability(self, beta, energy):
        """Compute the probability estimates using the sigmoid transformation."""
        dot = beta @ energy
        prob_sigma = self._sigmoid_transform(dot)
        return prob_sigma

    def draw_samples(self, n_samples=1000, seed=None):
        """Draw samples from the trained model.

        Args:
            n_samples (int, optional): Number of samples to draw. Defaults to 1000.
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        Raises:
            ValueError: If the model has not been fitted yet.

        Returns:
            np.ndarray: Samples drawn from the model.
        """

        if self.prob_estimates is None or self.model_params is None:
            raise ValueError("Model has not been fitted yet. Call train first.")

        # Generate random samples based on the last probabilities
        np.random.seed(seed)
        indices = np.random.randint(self.model_params[0].shape[0], size=n_samples)

        sample_betas = self.model_params[0][indices]
        prob_sigma = self._sigma_probability(sample_betas, self.model_params[1])

        samples = np.random.binomial(1, prob_sigma)
        return samples
