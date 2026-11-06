"""Abstract Base Class for machine learning model training strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


class BaseModelTrainer(ABC):
    """Abstract interface defining the contract for model trainer strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier of the model."""
        pass

    @property
    @abstractmethod
    def problem_type(self) -> str:
        """Target problem domain ('classification' or 'regression')."""
        pass

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaseModelTrainer":
        """Fit the model to the training dataset.

        Args:
            X_train: Feature matrix.
            y_train: Ground truth target series.

        Returns:
            Self instance.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point predictions for given feature matrix.

        Args:
            X: Feature matrix.

        Returns:
            Predicted target values array.
        """
        pass

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate prediction probabilities for classification tasks if supported.

        Args:
            X: Feature matrix.

        Returns:
            Probability matrix or None.
        """
        return None

    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model against unseen test data using domain-standard metrics.

        Args:
            X_test: Test feature matrix.
            y_test: True target series for test set.

        Returns:
            Dictionary mapping metric names to their scalar scores.
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return hyperparameters configured for this model instance."""
        pass
