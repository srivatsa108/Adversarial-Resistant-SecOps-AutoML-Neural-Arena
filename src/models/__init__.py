"""Models package: supervised trainers, autoencoders, and AutoML arena."""

from src.models.arena import ArenaResult, AutoMLArena
from src.models.autoencoder import AnomalyDetectionResult, DeepAutoencoderTrainer
from src.models.supervised import (
    DeepMLPClassifierTrainer,
    DeepMLPRegressorTrainer,
    LinearRegressionTrainer,
    LogisticRegressionTrainer,
    RandomForestClassifierTrainer,
    RandomForestRegressorTrainer,
    XGBoostClassifierTrainer,
)

__all__ = [
    "AnomalyDetectionResult",
    "ArenaResult",
    "AutoMLArena",
    "DeepAutoencoderTrainer",
    "DeepMLPClassifierTrainer",
    "DeepMLPRegressorTrainer",
    "LinearRegressionTrainer",
    "LogisticRegressionTrainer",
    "RandomForestClassifierTrainer",
    "RandomForestRegressorTrainer",
    "XGBoostClassifierTrainer",
]
