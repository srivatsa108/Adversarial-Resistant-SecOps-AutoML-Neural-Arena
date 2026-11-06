"""Configuration settings and security constraints for the AutoML platform."""

from dataclasses import dataclass, field
from typing import FrozenSet, Set


@dataclass(frozen=True)
class SecurityConfig:
    """Security configurations enforcing boundary defenses and safe processing."""

    # Maximum allowed file upload size (50 Megabytes)
    MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024

    # Permitted file extensions
    ALLOWED_EXTENSIONS: FrozenSet[str] = frozenset({"csv", "json", "pdf"})

    # Permitted MIME types
    ALLOWED_MIME_TYPES: FrozenSet[str] = frozenset(
        {
            "text/csv",
            "text/plain",
            "application/json",
            "application/pdf",
            "application/x-pdf",
            "application/octet-stream",  # Fallback often provided by web clients
        }
    )

    # Magic byte prefixes for file format signature verification
    MAGIC_SIGNATURES: dict[str, list[bytes]] = field(
        default_factory=lambda: {
            "pdf": [b"%PDF-"],
            "json": [b"{", b"[", b" \t\r\n{", b" \t\r\n["],
        }
    )

    # Regex patterns considered dangerous for Formula / CSV Injection
    FORMULA_PREFIXES: FrozenSet[str] = frozenset({"=", "+", "-", "@", "\t", "\r", "|"})

    # Threshold for continuous regression vs discrete classification
    # If integer/numeric unique value count <= CLASSIFICATION_CARDINALITY_THRESHOLD,
    # or if target is string/boolean/categorical, classification is triggered.
    CLASSIFICATION_CARDINALITY_THRESHOLD: int = 20

    # Near-zero variance threshold for junk column pruning
    DEFAULT_VARIANCE_THRESHOLD: float = 1e-4

    # Default random seed for reproducibility
    DEFAULT_RANDOM_STATE: int = 42


# Global default configuration instance
GLOBAL_CONFIG = SecurityConfig()
