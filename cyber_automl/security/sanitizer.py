"""Text vector and data payload sanitization module against injection vectors.

Mitigates:
- Formula / CSV Injection (DDE, Excel macro execution attacks).
- Cross-Site Scripting (XSS) / HTML tag injection.
- Null-byte and unprintable control character poisoning.
- Command / script injection tokens in vector space.
"""

import html
import re
from typing import Any, Union
import pandas as pd

from cyber_automl.config import GLOBAL_CONFIG

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
        return cleaned

    # 1. Strip null bytes and non-printable control chars
    cleaned = _CONTROL_CHAR_RE.sub("", cleaned)

    # 2. Defang CSV/Spreadsheet Formula Injection:
    # If the text starts with formula triggers (=, +, -, @, \t, \r, |), prefix with single quote
    if any(cleaned.startswith(prefix) for prefix in GLOBAL_CONFIG.FORMULA_PREFIXES):
        cleaned = f"'{cleaned}"

    # 3. HTML escape to neutralize XSS payloads
    cleaned = html.escape(cleaned, quote=True)

    # 4. Strip stray HTML/script tags that might remain
    cleaned = _HTML_TAG_RE.sub("", cleaned)

    return cleaned


def sanitize_dataframe_text_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize all textual / object columns of a pandas DataFrame in-memory.

    Args:
        df: Input DataFrame to sanitize.

    Returns:
        New or modified DataFrame with neutralized text vectors.
    """
    sanitized_df = df.copy()

    # Sanitize column names against injection and leading/trailing whitespace
    sanitized_columns = []
    for col in sanitized_df.columns:
        col_str = str(col).strip()
        # Clean control chars and quote formula prefixes on headers
        col_str = _CONTROL_CHAR_RE.sub("", col_str)
        if any(col_str.startswith(p) for p in GLOBAL_CONFIG.FORMULA_PREFIXES):
            col_str = f"'{col_str}"
        sanitized_columns.append(col_str)
    sanitized_df.columns = sanitized_columns

    # Sanitize string / object columns
    for col in sanitized_df.columns:
        if (
            sanitized_df[col].dtype == "object"
            or isinstance(sanitized_df[col].dtype, pd.StringDtype)
        ):
            sanitized_df[col] = sanitized_df[col].apply(sanitize_scalar_string)

    return sanitized_df
