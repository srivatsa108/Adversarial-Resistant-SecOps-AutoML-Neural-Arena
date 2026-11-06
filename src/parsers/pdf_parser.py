"""PDF parser using pdfplumber and regex text-mining for structured cyber logs."""

import io
import re
from typing import Any, Dict, List, Optional
import pandas as pd

from src.interfaces.parser import BaseFileParser
from src.security.sanitizer import sanitize_dataframe_text_vectors


class PDFFileParser(BaseFileParser):
    """PDF parser with table extraction and regex text mining fallback."""

    # Cybersecurity key-value pattern: e.g. "Source_IP: 192.168.1.1", "CVSS_Score: 7.5"
    KV_PATTERN = re.compile(
        r"^(?P<key>[a-zA-Z0-9_\-\.\s]{2,40})[:=]\s*(?P<val>[^\n\r]+)$",
        re.MULTILINE,
    )

    # Delimited row pattern (pipe, comma, tab, or multi-space table rows)
    TABLE_DELIMITER_PATTERN = re.compile(r"\s{2,}|\t|\|")

    def parse(self, buffer: io.BytesIO, filename: str) -> pd.DataFrame:
        """Parse in-memory PDF buffer into a sanitized pandas DataFrame.

        First attempts to extract structured tables using pdfplumber.
        If no tables exist, extracts raw text and applies regex text-mining
        to detect tabular rows or structured cybersecurity log events.

        Args:
            buffer: In-memory byte stream.
            filename: Source file name.

        Returns:
            Sanitized pd.DataFrame.

        Raises:
            ValueError: If PDF cannot be read or no features can be extracted.
        """
        buffer.seek(0)
        table_dfs: List[pd.DataFrame] = []
        raw_text_chunks: List[str] = []

        # 1. Try pdfplumber for table extraction
        try:
            import pdfplumber

            buffer.seek(0)
            with pdfplumber.open(buffer) as pdf:
                for page in pdf.pages:
                    # Extract tables if present
                    tables = page.extract_tables()
                    for tbl in tables:
                        if tbl and len(tbl) > 1:
                            headers = [str(h).strip() if h is not None else f"col_{i}" for i, h in enumerate(tbl[0])]
                            rows = tbl[1:]
                            # Filter empty rows
                            valid_rows = [r for r in rows if any(cell is not None and str(cell).strip() for cell in r)]
                            if valid_rows:
                                t_df = pd.DataFrame(valid_rows, columns=headers)
                                table_dfs.append(t_df)

                    # Extract page text
                    p_text = page.extract_text()
                    if p_text:
                        raw_text_chunks.append(p_text)
        except Exception:
            # Fallback to pypdf if pdfplumber is unavailable or encoutered error
            try:
                from pypdf import PdfReader

                buffer.seek(0)
                reader = PdfReader(buffer)
                for page in reader.pages:
                    txt = page.extract_text()
                    if txt:
                        raw_text_chunks.append(txt)
            except Exception as e:
                raise ValueError(f"Failed to read PDF file '{filename}': {e}")

        # If pdfplumber successfully parsed structured tables, combine them
        if table_dfs:
            combined = pd.concat(table_dfs, ignore_index=True)
            if not combined.empty and len(combined.columns) > 1:
                return sanitize_dataframe_text_vectors(combined)

        # 2. Fallback: Regex text mining across raw text chunks
        full_text = "\n".join(raw_text_chunks).strip()
        if not full_text:
            raise ValueError(f"PDF '{filename}' contained no extractable text or tables.")

        df_mined = self._mine_structured_data_from_text(full_text)
        if df_mined.empty or len(df_mined.columns) == 0:
            raise ValueError(
                f"Could not extract structured tabular cybersecurity records from PDF '{filename}'."
            )

        return sanitize_dataframe_text_vectors(df_mined)

    def _mine_structured_data_from_text(self, text: str) -> pd.DataFrame:
        """Mine structured key-value events or delimited rows from raw PDF text."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # Strategy A: Check for delimited lines (e.g. pipe or multi-space separated logs)
        delimited_rows: List[List[str]] = []
        for line in lines:
            parts = [p.strip() for p in self.TABLE_DELIMITER_PATTERN.split(line) if p.strip()]
            if len(parts) >= 3:
                delimited_rows.append(parts)

        if len(delimited_rows) >= 3:
            # Determine most common column count
            counts = [len(r) for r in delimited_rows]
            from collections import Counter
            modal_cols = Counter(counts).most_common(1)[0][0]
            matched_rows = [r for r in delimited_rows if len(r) == modal_cols]
            if len(matched_rows) >= 2:
                headers = matched_rows[0]
                rows = matched_rows[1:]
                return pd.DataFrame(rows, columns=headers)

        # Strategy B: Repeated key-value block mining (e.g. log stanzas)
        records: List[Dict[str, str]] = []
        current_record: Dict[str, str] = {}

        for line in lines:
            kv_match = self.KV_PATTERN.match(line)
            if kv_match:
                k = kv_match.group("key").strip().replace(" ", "_")
                v = kv_match.group("val").strip()
                if k in current_record:
                    # New record block started
                    records.append(current_record)
                    current_record = {k: v}
                else:
                    current_record[k] = v

        if current_record:
            records.append(current_record)

        if records and len(records) >= 1:
            return pd.DataFrame(records)

        # Strategy C: Fallback single line log entry mining
        log_records: List[Dict[str, Any]] = []
        for idx, line in enumerate(lines):
            log_records.append({
                "log_index": idx + 1,
                "raw_event": line,
                "character_length": len(line),
                "token_count": len(line.split()),
            })

        return pd.DataFrame(log_records)
