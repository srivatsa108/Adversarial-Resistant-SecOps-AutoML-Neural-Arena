"""Abstract Base Class for cybersecurity data cleaning and junk filtering engines."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


class BaseDataCleaner(ABC):
    """Abstract interface defining data deduplication, junk pruning, and imputation."""

    @abstractmethod
    def clean(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform end-to-end cleaning, junk filtering, and dynamic imputation.

        Args:
            df: Raw input DataFrame.
            target_column: Optional target column protected from pruning.

        Returns:
            Tuple of (cleaned pd.DataFrame, metadata report dict).
        """
        pass
