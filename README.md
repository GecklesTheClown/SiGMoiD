# SiGMoiD

SiGMoiD is a statistical approach to modelling high-dimesnional binary data inspired by statistical physics. This package is based on the paper "SiGMoiD: A super-statistical generative model for binary data", please cite the following if you use this code in your work: 

Zhao X, Plata G, Dixit PD (2021) SiGMoiD: A super-statistical generative model for binary data. PLOS Computational Biology 17(8): e1009275. https://doi.org/10.1371/journal.pcbi.1009275

In this implementation, we leverage PyTorch for fast model fitting and inference. Additionaly, we provide a model selection framework based on Akaike Information Criterion (AIC). 

Please reach out if you have any questions or suggestions!

## Installation

SiGMoiD can be installed via pip,

```bash
pip install https://github.com/GecklesTheClown/SiGMoiD.git
```

or via uv (https://github.com/astral-sh/uv),

```bash
uv add https://github.com/GecklesTheClown/SiGMoiD.git
```

or PyPI (coming soon).

```bash
pip install sigmoid-py
```

## GPU Support
This package supports GPU acceleration via PyTorch. If you have a compatible GPU and the appropriate PyTorch version installed, the package will automatically utilize the GPU for model fitting and inference. By default, the package will only install the CPU version of PyTorch.

If you wish to use GPU acceleration, please install PyTorch with Cuda support. Instructions for installation can be found on the [PyTorch website](https://pytorch.org/get-started/locally/).

## Quick Start

Here is a quick example of how to use SiGMoiD to fit a model to binary data.

```python
import pandas as pd
from sigmoid.selector import Selector
from sigmoid.model import Model

# Load your binary data into a pandas DataFrame
data = pd.read_csv('your_binary_data.csv')

# Initialize the model selector
selector = Selector(data, seed=42)
# Select the best model based on AIC
selector.fit(k=range(1, 21), repeats=10)
model = selector.optimal

# Draw samples from the optimal fitted model
samples = model.draw_samples(n_samples=1000, seed=42)
```

## Reproducibility
To ensure reproducibility, you can set the random seed when initializing the Selector or Model classes. During the model selection process, each tested model is seeded, where each seed is generated from a provided master seed.


## Roadmap
- Add more model selection criteria (for example BIC, cross-validation with different metrics).
- Improve model selection computational efficiency.
- Add adaptive learning rate schedulers.
