"""Abstract Base Class for Adversarial Robustness and Data Poisoning Stress Testing."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class BaseAdversarialTester(ABC):
    """Abstract interface defining adversarial evasion and poisoning defense evaluations."""

    @abstractmethod
    def evaluate_evasion_robustness(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        epsilons: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Perform adversarial perturbation stress testing across perturbation intensities.

        Args:
            model: Trained classifier or regressor model.
            X_test: Unseen test feature set.
            y_test: Ground truth targets.
            epsilons: List of perturbation bounds to test.

        Returns:
            Dictionary containing:
                - 'decay_curve': List of dicts with epsilon, score, accuracy_drop,
                - 'evasion_resistance_score': 0-100% metric,
                - 'threat_severity': 'Critical', 'High', 'Medium', 'Low',
                - 'secops_warning': Plain-English threat analysis for analysts.
        """
        pass
