"""Abstract Base Class for Deep Neural Network architectures and Autoencoders."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class BaseNeuralNetEngine(ABC):
    """Abstract interface defining Deep Neural Network engines (MLP & Autoencoder)."""

    @property
    @abstractmethod
    def architecture_name(self) -> str:
        """Name of the neural network architecture."""
        pass

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 25,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> Dict[str, List[float]]:
        """Train the deep neural network and return training history curves.

        Returns:
            Dictionary containing 'train_loss' and optionally 'val_loss' histories.
        """
        pass

    @abstractmethod
    def get_loss_history(self) -> Dict[str, List[float]]:
        """Retrieve epoch-by-epoch training and validation loss curves."""
        pass
