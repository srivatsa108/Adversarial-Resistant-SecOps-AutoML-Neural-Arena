"""Abstract Base Class for file parsing strategies."""

from abc import ABC, abstractmethod
import io
import pandas as pd


class BaseFileParser(ABC):
    """Abstract interface defining the contract for in-memory file parsing."""

    @abstractmethod
    def parse(self, buffer: io.BytesIO, filename: str) -> pd.DataFrame:
        """Parse raw binary buffer into a structured pandas DataFrame.

        Args:
            buffer: In-memory byte stream of the file content.
            filename: Original name of the file.

        Returns:
            pd.DataFrame containing the parsed data.

        Raises:
            ValueError: If parsing fails or data is structurally invalid.
        """
        pass
