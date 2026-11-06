"""Shared evaluation metrics module adhering to DRY principles."""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate centralized classification performance metrics.

    Args:
        y_true: Ground truth binary or multiclass labels.
        y_pred: Predicted target class labels.
        y_prob: Optional prediction probability distribution array.

    Returns:
        Dictionary of metric scores.
    """
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # Macro F1 for imbalanced cyber threat detection
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # ROC-AUC calculation if probabilities are supplied
    if y_prob is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                # Binary classification
                prob_vec = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] >= 2 else y_prob
                metrics["roc_auc"] = float(roc_auc_score(y_true, prob_vec))
            elif n_classes > 2:
                # Multiclass classification (one-vs-rest)
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        except Exception:
            metrics["roc_auc"] = 0.5

    return metrics


def evaluate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Calculate centralized regression performance metrics.

    Args:
        y_true: Ground truth continuous target array.
        y_pred: Predicted target values.

    Returns:
        Dictionary of metric scores.
    """
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def compute_reconstruction_error(
    X_original: np.ndarray,
    X_reconstructed: np.ndarray,
) -> np.ndarray:
    """Compute row-wise Mean Squared Reconstruction Error for Autoencoder anomaly detection.

    Args:
        X_original: Original scaled input feature matrix.
        X_reconstructed: Autoencoder reconstructed feature matrix.

    Returns:
        1D array of reconstruction error MSE per row.
    """
    diff = X_original - X_reconstructed
    return np.mean(diff ** 2, axis=1)
