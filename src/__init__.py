"""SecOps AutoML & Deep Neural Network Platform root package."""

from src.adversarial.tester import AdversarialRobustnessTester
from src.cleaners.cleaner import DataCleaner
from src.config import GLOBAL_CONFIG, SecurityConfig
from src.mitre.generator import MitreRuleGenerator
from src.models.arena import AutoMLArena
from src.parsers.factory import ParserFactory
from src.security.privacy import PrivacyMasker
from src.security.sanitizer import sanitize_dataframe_text_vectors
from src.security.validator import BufferValidator
from src.xai.explainer import SHAPExplainer

__all__ = [
    "AdversarialRobustnessTester",
    "AutoMLArena",
    "BufferValidator",
    "DataCleaner",
    "GLOBAL_CONFIG",
    "MitreRuleGenerator",
    "ParserFactory",
    "PrivacyMasker",
    "SHAPExplainer",
    "SecurityConfig",
    "sanitize_dataframe_text_vectors",
]
