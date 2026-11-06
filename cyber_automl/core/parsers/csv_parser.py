"""CSV file parser operating entirely in-memory."""

import io
import pandas as pd

from cyber_automl.core.interfaces.parser import BaseFileParser
from cyber_automl.security.sanitizer import sanitize_dataframe_text_vectors


class CSVFileParser(BaseFileParser):
    """In-memory parser for comma-separated value payloads."""

    def parse(self, buffer: io.BytesIO, filename: str) -> pd.DataFrame:
        """Parse in-memory CSV buffer into sanitized DataFrame.

        Args:
            buffer: In-memory byte stream.
            filename: Source file name.

        Returns:
            Sanitized pd.DataFrame.

        Raises:
            ValueError: If parsing fails or CSV is empty.
        """
        buffer.seek(0)
        raw_bytes = buffer.read()
        buffer.seek(0)

        # Attempt decoding with UTF-8 first, fallback to latin-1
        decoded_text = None
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                decoded_text = raw_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if decoded_text is None:
            raise ValueError(f"Failed to decode CSV '{filename}' with supported encodings.")

        string_io = io.StringIO(decoded_text)
        try:
            # Let pandas infer delimiter or default to comma
            df = pd.read_csv(string_io, sep=None, engine="python")
        except Exception as e:
            # Retry with explicit comma
            string_io.seek(0)
            try:
                df = pd.read_csv(string_io, sep=",")
            except Exception as nested_e:
                raise ValueError(f"Error parsing CSV file '{filename}': {nested_e}") from e

        if df.empty:
            raise ValueError(f"Parsed CSV file '{filename}' contains no rows.")

        return sanitize_dataframe_text_vectors(df)
