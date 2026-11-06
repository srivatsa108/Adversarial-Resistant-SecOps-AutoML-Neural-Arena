"""In-memory universal log parser supporting Syslog, CEF, and key-value security logs."""

import io
import re
from typing import Any, Dict, List, Optional
import pandas as pd

from src.interfaces.parser import BaseFileParser
from src.security.sanitizer import sanitize_dataframe_text_vectors


class LogFileParser(BaseFileParser):
    """Parses raw security log lines (CEF, Syslog RFC3164/5424, and Key-Value logs) into tabular DataFrames."""

    # CEF Format: CEF:Version|Device Vendor|Device Product|Device Version|Device Event Class ID|Name|Severity|Extension
    CEF_HEADER_PATTERN = re.compile(
        r"^CEF:\s*(\d+)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|(.*)$"
    )

    # Standard RFC3164 Syslog: Oct 11 22:14:15 myhost sshd[12345]: Failed password for root from 1.2.3.4
    SYSLOG_PATTERN = re.compile(
        r"^(?:<(?P<pri>\d+)>)?(?P<timestamp>[A-Z][a-z]{2}\s+\d+\s+\d{2}:\d{2}:\d{2}|\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+(?P<host>[^\s:]+)\s+(?P<process>[^:\[\s]+)(?:\[(?P<pid>\d+)\])?:\s+(?P<message>.*)$"
    )

    # Key-Value pairs: key=value or key="quoted value"
    KV_PATTERN = re.compile(r'([a-zA-Z0-9_\-\.]+)=(?:"([^"]*)"|([^\s,]+))')

    # Extraction patterns for unformatted syslog messages
    IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    PORT_PATTERN = re.compile(r"\b(?:port|sport|dport)\s*[:=]?\s*(\d{1,5})\b", re.IGNORECASE)

    def parse(self, buffer: io.BytesIO, filename: str) -> pd.DataFrame:
        """Parse text log stream into structured DataFrame entirely in RAM."""
        buffer.seek(0)
        content = buffer.read().decode("utf-8", errors="replace")
        lines = [line.strip() for line in content.splitlines() if line.strip()]

        if not lines:
            raise ValueError(f"Log file '{filename}' is empty.")

        records: List[Dict[str, Any]] = []

        for idx, line in enumerate(lines[:10000]):  # Cap to 10,000 lines for rapid processing
            parsed_row: Dict[str, Any] = {"log_index": idx + 1}

            # 1. Try CEF Format
            cef_match = self.CEF_HEADER_PATTERN.match(line)
            if cef_match:
                parsed_row["log_type"] = "CEF"
                parsed_row["vendor"] = cef_match.group(2)
                parsed_row["product"] = cef_match.group(3)
                parsed_row["event_id"] = cef_match.group(5)
                parsed_row["event_name"] = cef_match.group(6)
                parsed_row["severity"] = cef_match.group(7)
                extension = cef_match.group(8)

                # Parse extension key-values
                for kv in self.KV_PATTERN.finditer(extension):
                    k = kv.group(1).lower()
                    v = kv.group(2) if kv.group(2) is not None else kv.group(3)
                    parsed_row[k] = v

                records.append(parsed_row)
                continue

            # 2. Try Standard Syslog Format
            syslog_match = self.SYSLOG_PATTERN.match(line)
            if syslog_match:
                parsed_row["log_type"] = "SYSLOG"
                gd = syslog_match.groupdict()
                parsed_row["timestamp"] = gd.get("timestamp") or ""
                parsed_row["host"] = gd.get("host") or ""
                parsed_row["process"] = gd.get("process") or ""
                parsed_row["pid"] = gd.get("pid") or ""
                msg = gd.get("message") or ""
                parsed_row["raw_message"] = msg

                # Extract key-values if present in message body
                kvs = self.KV_PATTERN.findall(msg)
                if kvs:
                    for k, v_quoted, v_raw in kvs:
                        val = v_quoted if v_quoted else v_raw
                        parsed_row[k.lower()] = val
                else:
                    # Extract IPs and ports heuristic from message
                    ips = self.IP_PATTERN.findall(msg)
                    if ips:
                        parsed_row["source_ip"] = ips[0]
                        if len(ips) > 1:
                            parsed_row["destination_ip"] = ips[1]
                    ports = self.PORT_PATTERN.findall(msg)
                    if ports:
                        parsed_row["port"] = int(ports[0])

                    # Action inference
                    if any(w in msg.lower() for w in ("failed", "failure", "denied", "drop", "block", "invalid")):
                        parsed_row["action"] = "BLOCKED"
                        parsed_row["status_code"] = 1
                    else:
                        parsed_row["action"] = "ALLOWED"
                        parsed_row["status_code"] = 0

                records.append(parsed_row)
                continue

            # 3. Fallback: Generic Key-Value Log Line
            kvs = self.KV_PATTERN.findall(line)
            if kvs:
                parsed_row["log_type"] = "KEY_VALUE"
                for k, v_quoted, v_raw in kvs:
                    val = v_quoted if v_quoted else v_raw
                    parsed_row[k.lower()] = val
                records.append(parsed_row)
                continue

            # 4. Raw unstructured text log
            parsed_row["log_type"] = "RAW_TEXT"
            parsed_row["message"] = line
            ips = self.IP_PATTERN.findall(line)
            if ips:
                parsed_row["source_ip"] = ips[0]
            records.append(parsed_row)

        df = pd.DataFrame(records)
        df.fillna("NONE", inplace=True)
        return sanitize_dataframe_text_vectors(df)
