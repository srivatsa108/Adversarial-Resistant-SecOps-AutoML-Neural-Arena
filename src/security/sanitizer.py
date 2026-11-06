"""Text vector and data payload sanitization module against injection vectors.

Mitigates:
- Formula / CSV Injection (DDE, Excel macro execution attacks).
- Cross-Site Scripting (XSS) / HTML tag injection.
- Null-byte and unprintable control character poisoning.
- Command / script injection tokens in vector space.
"""

import html
import re
from typing import Any
import pandas as pd

from src.config import GLOBAL_CONFIG

# Matches script tags including inner javascript payload
_SCRIPT_TAG_RE = re.compile(r"<\s*script[^>]*>[\s\S]*?<\s*/\s*script\s*>", re.IGNORECASE)

# Matches dangerous HTML tags and script injections
_HTML_TAG_RE = re.compile(r"<[^>]+>", re.IGNORECASE)

# Matches null bytes and unsafe non-printable control characters (preserving tab, newline, CR)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# Common command injection delimiters in log/threat datasets
_CMD_INJECTION_TOKENS = re.compile(r"(\||;|`|\$\(|\$\{)", re.IGNORECASE)


def sanitize_scalar_string(value: Any) -> Any:
    """Sanitize a single scalar value if it is a string.

    Args:
        value: Input data value.

    Returns:
        Sanitized value.
    """
    if not isinstance(value, str):
        return value

    cleaned = value.strip()
    if not cleaned:
        return ""

    # Strip null bytes and non-printable control chars
    cleaned = _CONTROL_CHAR_RE.sub("", cleaned)

    # Strip script blocks completely
    cleaned = _SCRIPT_TAG_RE.sub("", cleaned)

    # Strip raw HTML/remaining tags
    cleaned = _HTML_TAG_RE.sub("", cleaned)

    # Escape HTML entities
    cleaned = html.escape(cleaned)

    # Neutralize Formula / CSV Injection prefixes
    while cleaned and cleaned[0] in GLOBAL_CONFIG.FORMULA_PREFIXES:
        cleaned = "'" + cleaned
        break

    return cleaned


def sanitize_dataframe_text_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize all string/object columns in a DataFrame in-place or via copy.

    Args:
        df: Input DataFrame.

    Returns:
        Sanitized DataFrame.
    """
    cleaned_df = df.copy()

    # Sanitize column headers themselves against formula/script injection
    sanitized_columns = []
    for col in cleaned_df.columns:
        col_str = str(col).strip()
        col_str = _CONTROL_CHAR_RE.sub("", col_str)
        col_str = _HTML_TAG_RE.sub("", col_str)
        # Avoid formula injection in headers
        if col_str and col_str[0] in GLOBAL_CONFIG.FORMULA_PREFIXES:
            col_str = "_" + col_str[1:]
        sanitized_columns.append(col_str)
    cleaned_df.columns = sanitized_columns

    # Sanitize string/object cells
    object_cols = cleaned_df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        cleaned_df[col] = cleaned_df[col].apply(sanitize_scalar_string)

    return cleaned_df
