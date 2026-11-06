"""Security package: buffer validation, size limits, and vector sanitization."""

from cyber_automl.security.sanitizer import (
    sanitize_dataframe_text_vectors,
    sanitize_scalar_string,
)
from cyber_automl.security.validator import (
    BufferValidator,
    SecurityValidationError,
)

__all__ = [
    "BufferValidator",
    "SecurityValidationError",
    "sanitize_dataframe_text_vectors",
    "sanitize_scalar_string",
]
