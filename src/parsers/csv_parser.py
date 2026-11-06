"""In-memory CSV parser implementing BaseFileParser."""

import io
from typing import Optional
import pandas as pd

from src.interfaces.parser import BaseFileParser
from src.security.sanitizer import sanitize_dataframe_text_vectors


class CSVFileParser(BaseFileParser):
    """In-memory parser for CSV datasets with automatic delimiter detection and sanitization."""

    def parse(self, buffer: io.BytesIO, filename: str) -> pd.DataFrame:
        """Parse in-memory CSV buffer into a sanitized pandas DataFrame.

        Args:
            buffer: In-memory byte stream.
            filename: Source file name.

        Returns:
            Sanitized pd.DataFrame.

        Raises:
            ValueError: If file is empty or cannot be decoded.
        """
        buffer.seek(0)
        raw_bytes = buffer.read()
        buffer.seek(0)

        decoded_text: Optional[str] = None
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                decoded_text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if decoded_text is None:
            raise ValueError(f"Failed to decode CSV '{filename}' using supported character encodings.")

        if not decoded_text.strip():
            raise ValueError(f"CSV file '{filename}' contains no readable content.")

        string_io = io.StringIO(decoded_text)
        try:
            # Let pandas infer delimiter or default to comma
            df = pd.read_csv(string_io, sep=None, engine="python")
        except Exception as e:
            string_io.seek(0)
            try:
                df = pd.read_csv(string_io, sep=",")
            except Exception as inner_e:
                raise ValueError(f"Could not parse CSV content from '{filename}': {inner_e}") from e

        if df.empty or len(df.columns) == 0:
            raise ValueError(f"Parsed CSV '{filename}' yielded an empty dataset.")

        return sanitize_dataframe_text_vectors(df)
