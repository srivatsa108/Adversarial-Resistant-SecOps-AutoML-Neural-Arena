"""Abstract Base Class for Explainable AI (XAI) feature importance computation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd


class BaseExplainer(ABC):
    """Abstract interface defining Explainable AI (SHAP) engine for Deep Nets and ML."""

    @abstractmethod
    def explain_global(
        self,
        model: Any,
        X_sample: pd.DataFrame,
        max_display: int = 10,
    ) -> Dict[str, Any]:
        """Compute global feature importances using SHAP values.

        Returns:
            Dictionary with feature names, mean absolute SHAP values, and plain-English summaries.
        """
        pass

    @abstractmethod
    def explain_instance(
        self,
        model: Any,
        instance: pd.Series,
        background_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute local feature attribution for a single suspicious log event or anomaly.

        Returns:
            Dictionary mapping feature contributions to root causes in plain English.
        """
        pass
