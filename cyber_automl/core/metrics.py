"""Shared evaluation metrics module adhering to the DRY principle.

Provides centralized, reusable calculation functions for both classification
and regression evaluation suites.
"""

from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


class EvaluationMetrics:
    """Centralized calculator for classification and regression performance metrics."""

    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate standard evaluation metrics for classification models.

        Computes:
            - accuracy: Proportion of correct classifications
            - f1_score: Weighted F1 score (resilient to class imbalance in cyber alerts)
            - f1_macro: Unweighted mean of F1 scores across classes
            - precision: Weighted precision (zero_division safe)
            - recall: Weighted recall (zero_division safe)

        Args:
            y_true: True ground truth class labels.
            y_pred: Predicted class labels.

        Returns:
            Dictionary of calculated scalar metrics.
        """
        acc = float(accuracy_score(y_true, y_pred))
        f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        prec_w = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec_w = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

        return {
            "accuracy": round(acc, 4),
            "f1_score": round(f1_w, 4),
            "f1_macro": round(f1_m, 4),
            "precision": round(prec_w, 4),
            "recall": round(rec_w, 4),
        }

    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate standard evaluation metrics for continuous regression models.

        Computes:
            - rmse: Root Mean Squared Error
            - r_squared: Coefficient of Determination (R²)
            - mae: Mean Absolute Error

        Args:
            y_true: True continuous targets.
            y_pred: Predicted continuous values.

        Returns:
            Dictionary of calculated scalar metrics.
        """
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))

        return {
            "rmse": round(rmse, 4),
            "r_squared": round(r2, 4),
            "mae": round(mae, 4),
        }
