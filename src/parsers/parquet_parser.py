"""In-memory Parquet file parser supporting enterprise security data lakes."""

import io
import pandas as pd

from src.interfaces.parser import BaseFileParser
from src.security.sanitizer import sanitize_dataframe_text_vectors


class ParquetFileParser(BaseFileParser):
    """In-memory parser for columnar Parquet security telemetry."""

    def parse(self, buffer: io.BytesIO, filename: str) -> pd.DataFrame:
        """Parse Parquet byte buffer directly into a DataFrame in RAM."""
        buffer.seek(0)
        try:
            df = pd.read_parquet(buffer)
            return sanitize_dataframe_text_vectors(df)
        except Exception as e:
            raise ValueError(f"Failed to parse Parquet file '{filename}': {e}")
