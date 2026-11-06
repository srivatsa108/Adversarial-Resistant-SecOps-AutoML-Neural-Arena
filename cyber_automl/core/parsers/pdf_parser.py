"""PDF file parser with fallback text mining for cybersecurity data in-memory."""

import io
import re
from typing import Any, Dict, List, Optional
import pandas as pd
from pypdf import PdfReader

from cyber_automl.core.interfaces.parser import BaseFileParser
from cyber_automl.security.sanitizer import sanitize_dataframe_text_vectors


class PDFFileParser(BaseFileParser):
    """PDF parser with structured table extraction and regex text mining fallback."""

    # Common cybersecurity key-value pattern: e.g. "Host: 192.168.1.1", "CVSS_Score: 7.5"
    KV_PATTERN = re.compile(
        r"^(?P<key>[a-zA-Z0-9_\-\.\s]{2,40})[:=]\s*(?P<val>[^\n\r]+)$",
        re.MULTILINE,
    )

    # Delimited row pattern (pipe, tab, or multi-space table rows)
    TABLE_ROW_PATTERN = re.compile(r"\s{2,}|\t|\|")

    def parse(self, buffer: io.BytesIO, filename: str) -> pd.DataFrame:
        """Parse in-memory PDF buffer into sanitized DataFrame.

        Extracts raw text and attempts structured tabular parsing. If tabular parsing
        fails, triggers fallback text mining to extract structured key-value spaces
        or regex-based log arrays.

        Args:
            buffer: In-memory byte stream.
            filename: Source file name.

        Returns:
            Sanitized pd.DataFrame.

        Raises:
            ValueError: If PDF cannot be read or no structured features can be extracted.
        """
        buffer.seek(0)
        try:
            reader = PdfReader(buffer)
            extracted_pages_text: List[str] = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    extracted_pages_text.append(page_text)
        except Exception as e:
            raise ValueError(f"Failed to read PDF buffer '{filename}': {e}") from e

        full_text = "\n".join(extracted_pages_text).strip()
        if not full_text:
            raise ValueError(f"PDF file '{filename}' contains no extractable text.")

        # Strategy 1: Attempt delimited table parsing
        df = self._attempt_table_parse(full_text)
        if df is not None and len(df) >= 2 and len(df.columns) >= 2:
            return sanitize_dataframe_text_vectors(df)

        # Strategy 2: Fallback text mining for repeated key-value entities or log events
        df = self._mine_structured_records(full_text)
        if df is not None and not df.empty:
            return sanitize_dataframe_text_vectors(df)

        # Strategy 3: Regex array extraction for common security entities
        df = self._mine_regex_arrays(full_text)
        if df is not None and not df.empty:
            return sanitize_dataframe_text_vectors(df)

        raise ValueError(
            f"Unable to extract structured tabular or key-value data from PDF '{filename}'."
        )

    def _attempt_table_parse(self, text: str) -> Optional[pd.DataFrame]:
        """Attempt to extract structured rows delimited by pipes, tabs, or aligned whitespace."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return None

        # Check for pipe-delimited table (e.g., Markdown or text report tables)
        pipe_lines = [l for l in lines if "|" in l]
        if len(pipe_lines) >= 3:
            try:
                table_str = "\n".join(pipe_lines)
                df = pd.read_csv(io.StringIO(table_str), sep="|", engine="python")
                # Drop uninformative empty columns created by leading/trailing pipes
                df = df.dropna(axis=1, how="all")
                df.columns = [c.strip() for c in df.columns]
                if not df.empty and len(df.columns) >= 2:
                    return df
            except Exception:
                pass

        # Check for tab or multi-space delimited rows
        parsed_rows: List[List[str]] = []
        for line in lines:
            parts = [p.strip() for p in self.TABLE_ROW_PATTERN.split(line) if p.strip()]
            if len(parts) >= 2:
                parsed_rows.append(parts)

        if len(parsed_rows) >= 3:
            # Check if columns count is consistent
            col_counts = [len(r) for r in parsed_rows]
            most_common_len = max(set(col_counts), key=col_counts.count)
            filtered_rows = [r for r in parsed_rows if len(r) == most_common_len]

            if len(filtered_rows) >= 3:
                header = filtered_rows[0]
                data = filtered_rows[1:]
                # Verify header items look like column names
                if len(set(header)) == len(header):
                    return pd.DataFrame(data, columns=header)

        return None

    def _mine_structured_records(self, text: str) -> Optional[pd.DataFrame]:
        """Mine multi-record key-value blocks (e.g., incident reports, vulnerability records)."""
        # Split text into potential blocks or records (delimited by blank lines or headers)
        blocks = re.split(r"\n\s*\n|---|\b(?:Record|Event|Vulnerability|Incident|Host)\s*#?\d+[:\-]?", text)
        records: List[Dict[str, Any]] = []

        for block in blocks:
            matches = self.KV_PATTERN.findall(block)
            if len(matches) >= 2:
                record = {k.strip().replace(" ", "_"): v.strip() for k, v in matches}
                records.append(record)

        if len(records) >= 2:
            df = pd.DataFrame(records)
            # Try to convert numeric columns
            return df.apply(pd.to_numeric, errors="ignore")

        # If only one block of key-values was found across entire document
        all_matches = self.KV_PATTERN.findall(text)
        if len(all_matches) >= 3:
            # Check for recurring keys which indicates sequential events
            keys = [k.strip().replace(" ", "_") for k, _ in all_matches]
            first_key = keys[0]
            if keys.count(first_key) > 1:
                # Group by first recurring key
                grouped_records: List[Dict[str, Any]] = []
                current_record: Dict[str, Any] = {}
                for k, v in all_matches:
                    clean_k = k.strip().replace(" ", "_")
                    clean_v = v.strip()
                    if clean_k == first_key and current_record:
                        grouped_records.append(current_record)
                        current_record = {}
                    current_record[clean_k] = clean_v
                if current_record:
                    grouped_records.append(current_record)
                if len(grouped_records) >= 2:
                    return pd.DataFrame(grouped_records).apply(pd.to_numeric, errors="ignore")

        return None

    def _mine_regex_arrays(self, text: str) -> Optional[pd.DataFrame]:
        """Fallback regex miner extracting arrays of IPs, ports, severity labels, and metrics."""
        ip_pattern = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
        port_pattern = re.compile(r"\bport\s*[:=]?\s*([0-9]{1,5})\b", re.IGNORECASE)
        severity_pattern = re.compile(r"\b(CRITICAL|HIGH|MEDIUM|LOW|INFO|INFORMATIONAL)\b", re.IGNORECASE)
        score_pattern = re.compile(r"\b(?:cvss|score|metric|risk)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)

        ips = ip_pattern.findall(text)
        ports = port_pattern.findall(text)
        severities = severity_pattern.findall(text)
        scores = score_pattern.findall(text)

        # If we have consistent lengths or can align arrays into security events
        min_len = min(len(ips), len(severities)) if ips and severities else 0
        if min_len >= 2:
            data = {
                "source_ip": ips[:min_len],
                "severity": [s.upper() for s in severities[:min_len]],
            }
            if len(ports) >= min_len:
                data["port"] = [int(p) for p in ports[:min_len]]
            if len(scores) >= min_len:
                data["risk_score"] = [float(s) for s in scores[:min_len]]
            return pd.DataFrame(data)

        return None
