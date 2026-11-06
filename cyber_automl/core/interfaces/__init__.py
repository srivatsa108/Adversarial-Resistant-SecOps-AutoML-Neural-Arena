"""Core abstract interfaces for parsers and trainers."""

from cyber_automl.core.interfaces.parser import BaseFileParser
from cyber_automl.core.interfaces.trainer import BaseModelTrainer

__all__ = ["BaseFileParser", "BaseModelTrainer"]
