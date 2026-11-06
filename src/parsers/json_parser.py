"""In-memory JSON parser implementing BaseFileParser."""

import io
import json
from typing import Any, Dict, List
import pandas as pd

from src.interfaces.parser import BaseFileParser
from src.security.sanitizer import sanitize_dataframe_text_vectors


class JSONFileParser(BaseFileParser):
    """In-memory parser for JSON logs, arrays of objects, or line-delimited JSON."""

    def parse(self, buffer: io.BytesIO, filename: str) -> pd.DataFrame:
        """Parse in-memory JSON buffer into a sanitized pandas DataFrame.

        Args:
            buffer: In-memory byte stream.
            filename: Source file name.

        Returns:
            Sanitized pd.DataFrame.

        Raises:
            ValueError: If JSON is invalid or cannot be converted to a tabular DataFrame.
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

        data: Any = None
        # Try standard JSON parsing first
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback to JSON Lines (ndjson)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            records: List[Dict[str, Any]] = []
            for line_idx, line in enumerate(lines):
                try:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        records.append(record)
                except json.JSONDecodeError:
                    continue
            if records:
                data = records
            else:
                raise ValueError(f"Unable to parse '{filename}' as standard JSON or JSON Lines.")

        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            # Check for common envelope keys like {"data": [...], "logs": [...], "events": [...]}
            nested_keys = ["data", "logs", "events", "records", "alerts", "rows"]
            target_list = None
            for key in nested_keys:
                if key in data and isinstance(data[key], list):
                    target_list = data[key]
                    break

            if target_list is not None:
                df = pd.json_normalize(target_list)
            else:
                # Single record or dictionary of arrays
                try:
                    df = pd.DataFrame(data)
                except ValueError:
                    df = pd.json_normalize([data])
        else:
            raise ValueError(f"JSON root structure in '{filename}' is neither a list nor an object.")

        if df.empty or len(df.columns) == 0:
            raise ValueError(f"Parsed JSON '{filename}' yielded zero tabular records.")

        return sanitize_dataframe_text_vectors(df)
