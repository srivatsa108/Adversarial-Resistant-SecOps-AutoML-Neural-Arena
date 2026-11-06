"""In-memory buffer validation and file constraint enforcer.

Ensures strict file size verification, extension validation, and magic byte
inspection entirely in-memory without persistent disk interaction.
"""

import io
from typing import Optional

from src.config import GLOBAL_CONFIG, SecurityConfig


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
            Normalized file extension ('csv', 'json', 'pdf').

        Raises:
            SecurityValidationError: If the buffer violates security constraints.
        """
        if not filename or not isinstance(filename, str):
            raise SecurityValidationError("Filename cannot be empty or invalid.")

        # Check extension
        parts = filename.rsplit(".", 1)
        if len(parts) < 2:
            raise SecurityValidationError(f"File '{filename}' lacks a valid file extension.")

        ext = parts[1].lower().strip()
        if ext not in self._config.ALLOWED_EXTENSIONS:
            raise SecurityValidationError(
                f"Extension '.{ext}' is disallowed. Allowed: {sorted(self._config.ALLOWED_EXTENSIONS)}"
            )

        # Check MIME type if supplied
        if content_type:
            cleaned_mime = content_type.split(";")[0].strip().lower()
            if cleaned_mime not in self._config.ALLOWED_MIME_TYPES:
                raise SecurityValidationError(
                    f"MIME type '{cleaned_mime}' is not permitted for processing."
                )

        # Validate buffer length without disk persistence
        buffer.seek(0, io.SEEK_END)
        size = buffer.tell()
        buffer.seek(0)

        if size == 0:
            raise SecurityValidationError(f"Uploaded file '{filename}' is empty (0 bytes).")

        if size > self._config.MAX_FILE_SIZE_BYTES:
            max_mb = self._config.MAX_FILE_SIZE_BYTES / (1024 * 1024)
            actual_mb = size / (1024 * 1024)
            raise SecurityValidationError(
                f"File size exceeds safety quota: {actual_mb:.2f} MB > limit of {max_mb:.0f} MB."
            )

        # Magic byte signature verification
        if ext in self._config.MAGIC_SIGNATURES:
            buffer.seek(0)
            header = buffer.read(32)
            buffer.seek(0)
            valid_signatures = self._config.MAGIC_SIGNATURES[ext]
            matched = any(header.startswith(sig.strip()) or sig in header for sig in valid_signatures)
            if not matched:
                raise SecurityValidationError(
                    f"Magic byte header verification failed for .{ext} file: '{filename}'."
                )

        return ext
