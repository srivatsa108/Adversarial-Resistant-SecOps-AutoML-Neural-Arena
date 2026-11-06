"""JSON file parser operating entirely in-memory."""

import io
import json
import pandas as pd

from cyber_automl.core.interfaces.parser import BaseFileParser
from cyber_automl.security.sanitizer import sanitize_dataframe_text_vectors


class JSONFileParser(BaseFileParser):
    """In-memory parser for JSON records, arrays, or line-delimited JSON."""

    def parse(self, buffer: io.BytesIO, filename: str) -> pd.DataFrame:
        """Parse in-memory JSON buffer into sanitized DataFrame.

        Args:
            buffer: In-memory byte stream.
            filename: Source file name.

        Returns:
            Sanitized pd.DataFrame.

        Raises:
            ValueError: If parsing fails or JSON structure is invalid.
        """
        buffer.seek(0)
        raw_bytes = buffer.read()
        buffer.seek(0)

        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1")

        text = text.strip()
        if not text:
            raise ValueError(f"JSON file '{filename}' is empty.")

        try:
            # Try parsing as standard JSON
            parsed_data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: check for newline-delimited JSON (NDJSON/JSONL) common in cyber logs
            try:
                df = pd.read_json(io.StringIO(text), lines=True)
                if not df.empty:
                    return sanitize_dataframe_text_vectors(df)
            except Exception as e:
                raise ValueError(f"Error parsing JSON payload in '{filename}': {e}")

        # Normalize structured dict/list into DataFrame
        if isinstance(parsed_data, list):
            df = pd.json_normalize(parsed_data)
        elif isinstance(parsed_data, dict):
            # Check if dict wraps a list (e.g., {"events": [...]}, {"records": [...]}, {"data": [...]})
            common_keys = ("data", "events", "records", "items", "rows", "logs")
            found_key = next((k for k in common_keys if k in parsed_data and isinstance(parsed_data[k], list)), None)
            if found_key:
                df = pd.json_normalize(parsed_data[found_key])
            else:
                # Single row dict or dict of columns
                try:
                    df = pd.DataFrame(parsed_data)
                except ValueError:
                    df = pd.json_normalize([parsed_data])
        else:
            raise ValueError(f"Unsupported JSON root type: {type(parsed_data)}")

        if df.empty:
            raise ValueError(f"Parsed JSON file '{filename}' yielded an empty dataset.")

        return sanitize_dataframe_text_vectors(df)
