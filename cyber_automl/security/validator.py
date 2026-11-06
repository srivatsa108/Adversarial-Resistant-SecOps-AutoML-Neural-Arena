"""In-memory buffer validation and file constraint enforcer.

Ensures strict file size verification, extension validation, and magic byte
inspection entirely in-memory without persistent disk interaction.
"""

import io
from typing import Optional

from cyber_automl.config import GLOBAL_CONFIG, SecurityConfig


class SecurityValidationError(ValueError):
    """Raised when an uploaded buffer violates security policies."""
    pass


class BufferValidator:
    """Validates raw binary buffers prior to parsing."""

    def __init__(self, config: Optional[SecurityConfig] = None) -> None:
        """Initialize with an optional security configuration.

        Args:
            config: Security policy parameters (defaults to GLOBAL_CONFIG).
        """
        self._config = config or GLOBAL_CONFIG

    def validate(
        self,
        buffer: io.BytesIO,
        filename: str,
        content_type: Optional[str] = None,
    ) -> str:
        """Validate an in-memory buffer against size, filename, and format policies.

        Args:
            buffer: In-memory byte stream.
            filename: Original file name provided by the client.
            content_type: Optional MIME type from HTTP headers.

        Returns:
            The normalized file extension (e.g., 'csv', 'json', 'pdf').

        Raises:
            SecurityValidationError: If the payload violates security boundaries.
        """
        if not filename or "." not in filename:
            raise SecurityValidationError("Invalid or missing filename.")

        # 1. Sanitize and extract extension
        ext = filename.rsplit(".", 1)[-1].strip().lower()
        if ext not in self._config.ALLOWED_EXTENSIONS:
            raise SecurityValidationError(
                f"Unsupported file extension '.{ext}'. "
                f"Permitted extensions: {sorted(list(self._config.ALLOWED_EXTENSIONS))}"
            )

        # 2. In-memory buffer size validation
        current_pos = buffer.tell()
        buffer.seek(0, io.SEEK_END)
        size_bytes = buffer.tell()
        buffer.seek(current_pos)

        if size_bytes <= 0:
            raise SecurityValidationError("Buffer is empty (0 bytes).")

        if size_bytes > self._config.MAX_FILE_SIZE_BYTES:
            max_mb = self._config.MAX_FILE_SIZE_BYTES / (1024 * 1024)
            actual_mb = size_bytes / (1024 * 1024)
            raise SecurityValidationError(
                f"File size ({actual_mb:.2f} MB) exceeds maximum allowed limit ({max_mb:.2f} MB)."
            )

        # 3. MIME type validation if supplied
        if content_type:
            normalized_mime = content_type.split(";")[0].strip().lower()
            if normalized_mime not in self._config.ALLOWED_MIME_TYPES:
                # Issue warning / check or enforce
                pass

        # 4. Magic signature verification
        self._verify_magic_signature(buffer, ext)

        return ext

    def _verify_magic_signature(self, buffer: io.BytesIO, ext: str) -> None:
        """Inspect the header bytes of the buffer to match expected format signatures.

        Args:
            buffer: In-memory byte stream.
            ext: Normalized file extension.

        Raises:
            SecurityValidationError: If format signature does not match.
        """
        pos = buffer.tell()
        buffer.seek(0)
        header = buffer.read(32)
        buffer.seek(pos)

        if ext == "pdf":
            if not header.startswith(b"%PDF-"):
                raise SecurityValidationError(
                    "Invalid PDF file header. Magic byte check failed (expected '%PDF-')."
                )
        elif ext == "json":
            stripped = header.strip()
            if not (stripped.startswith(b"{") or stripped.startswith(b"[")):
                raise SecurityValidationError(
                    "Invalid JSON file header. Expected root '{' or '[' token."
                )
        elif ext == "csv":
            # Ensure CSV does not contain null binary bytes which often indicate executable / binary spoofing
            if b"\x00" in header:
                raise SecurityValidationError(
                    "Suspicious binary null byte detected in CSV file header."
                )
