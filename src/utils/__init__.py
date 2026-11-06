"""Utilities package: metrics, translations, and plotting."""

from src.utils.metrics import (
    compute_reconstruction_error,
    evaluate_classification_metrics,
    evaluate_regression_metrics,
)
from src.utils.plotting import (
    create_adversarial_decay_curve,
    create_anomaly_reconstruction_plot,
    create_model_comparison_radar_chart,
    create_neural_loss_curve,
    create_shap_feature_importance_bar,
)
from src.utils.translations import (
    get_threat_badge,
    translate_adversarial_to_secops_warning,
    translate_metrics_to_secops,
    translate_shap_to_secops_root_causes,
)

__all__ = [
    "compute_reconstruction_error",
    "create_adversarial_decay_curve",
    "create_anomaly_reconstruction_plot",
    "create_model_comparison_radar_chart",
    "create_neural_loss_curve",
    "create_shap_feature_importance_bar",
    "evaluate_classification_metrics",
    "evaluate_regression_metrics",
    "get_threat_badge",
    "translate_adversarial_to_secops_warning",
    "translate_metrics_to_secops",
    "translate_shap_to_secops_root_causes",
]
