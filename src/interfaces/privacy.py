"""Abstract Base Class for Zero-Trust Privacy and Cryptographic Anonymization."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import pandas as pd


class BasePrivacyEngine(ABC):
    """Abstract interface defining Zero-Trust local masking and de-masking."""

    @abstractmethod
    def mask_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scan and mask sensitive entities (IPv4, IPv6, MAC, emails, sensitive URLs) in a DataFrame.

        Args:
            df: Input DataFrame containing potential PII or internal infrastructure metadata.

        Returns:
            Tuple of (masked pd.DataFrame, masking audit statistics).
        """
        pass

    @abstractmethod
    def mask_text(self, text: str) -> str:
        """Scan and mask sensitive entities in an arbitrary text string.

        Args:
            text: Raw input text.

        Returns:
            Anonymized text with internal entities replaced with synthetic cryptographic tokens.
        """
        pass

    @abstractmethod
    def demask_text(self, text: str) -> str:
        """Restore masked synthetic tokens to real entities locally on the user interface.

        Args:
            text: Text containing masked tokens (e.g., from an AI agent).

        Returns:
            Original human-readable text strictly for local presentation.
        """
        pass

    @abstractmethod
    def demask_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively demask dictionaries or JSON payloads locally.

        Args:
            data: Masked dictionary.

        Returns:
            Demasked dictionary.
        """
        pass
