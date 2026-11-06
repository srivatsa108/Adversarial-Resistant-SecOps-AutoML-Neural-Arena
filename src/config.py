"""Configuration settings, security thresholds, and system constants for the SecOps AutoML Platform."""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List


@dataclass(frozen=True)
class SecurityConfig:
    """Security configurations enforcing boundary defenses and safe in-memory processing."""

    # Maximum permitted file upload size (1 Gigabyte)
    MAX_FILE_SIZE_BYTES: int = 1024 * 1024 * 1024

    # Permitted file extensions
    ALLOWED_EXTENSIONS: FrozenSet[str] = frozenset({"csv", "json", "pdf", "pcap", "pcapng"})

    # Permitted MIME types
    ALLOWED_MIME_TYPES: FrozenSet[str] = frozenset(
        {
            "text/csv",
            "text/plain",
            "application/json",
            "application/pdf",
            "application/x-pdf",
            "application/vnd.tcpdump.pcap",
            "application/x-pcap",
            "application/pcap",
            "application/octet-stream",
        }
    )

    # Magic byte signatures for strict binary validation
    MAGIC_SIGNATURES: Dict[str, List[bytes]] = field(
        default_factory=lambda: {
            "pdf": [b"%PDF-"],
            "json": [b"{", b"[", b" \t\r\n{", b" \t\r\n["],
            "pcap": [
                b"\xd4\xc3\xb2\xa1",
                b"\xa1\xb2\xc3\xd4",
                b"\x4d\x3c\xb2\xa1",
                b"\xa1\xb2\x3c\x4d",
            ],
            "pcapng": [b"\x0a\x0d\x0d\x0a"],
        }
    )

    # Formula / CSV Injection dangerous starting characters
    FORMULA_PREFIXES: FrozenSet[str] = frozenset({"=", "+", "-", "@", "\t", "\r", "|"})

    # Max cardinality for automatic classification vs regression detection
    CLASSIFICATION_CARDINALITY_THRESHOLD: int = 20

    # Near-zero variance threshold for junk metadata pruning
    DEFAULT_VARIANCE_THRESHOLD: float = 1e-4

    # Default random seed for deterministic reproducibility
    DEFAULT_RANDOM_STATE: int = 42

    # Default contamination rate for unsupervised Autoencoder anomaly detection (5%)
    DEFAULT_CONTAMINATION_RATE: float = 0.05

    # Adversarial test perturbation epsilons (normalized feature space)
    DEFAULT_ADVERSARIAL_EPSILONS: List[float] = field(
        default_factory=lambda: [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    )

    # Default LLM model identifier for MITRE rule generation
    DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"


# Global singleton configuration instance
GLOBAL_CONFIG = SecurityConfig()
