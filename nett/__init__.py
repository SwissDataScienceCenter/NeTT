"""NeTT -- Neural Tensor Tomography: implicit neural fields for X-ray scattering tensor tomography."""
from .config import NeTTConfig
from .train import FitResult, fit

__version__ = "0.1.0"
__all__ = ["NeTTConfig", "fit", "FitResult", "__version__"]
