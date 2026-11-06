"""Parser factory for resolving file parsers based on format."""

from typing import Dict, Type
from cyber_automl.core.interfaces.parser import BaseFileParser
from cyber_automl.core.parsers.csv_parser import CSVFileParser
from cyber_automl.core.parsers.json_parser import JSONFileParser
from cyber_automl.core.parsers.pdf_parser import PDFFileParser


class ParserFactory:
    """Factory resolving appropriate parser strategies for supported file extensions."""

    _registry: Dict[str, Type[BaseFileParser]] = {
        "csv": CSVFileParser,
        "json": JSONFileParser,
        "pdf": PDFFileParser,
    }

    @classmethod
    def register_parser(cls, extension: str, parser_cls: Type[BaseFileParser]) -> None:
        """Register a new parser strategy.

        Args:
            extension: Lowercase file extension without dot.
            parser_cls: Subclass of BaseFileParser.
        """
        cls._registry[extension.lower()] = parser_cls

    @classmethod
    def get_parser(cls, extension: str) -> BaseFileParser:
        """Instantiate and return the parser matching the file extension.

        Args:
            extension: File extension (e.g., 'csv', 'json', 'pdf').

        Returns:
            An instance of BaseFileParser.

        Raises:
            ValueError: If the file extension is not supported.
        """
        norm_ext = extension.lower().strip()
        parser_cls = cls._registry.get(norm_ext)
        if not parser_cls:
            raise ValueError(
                f"No parser registered for file extension '{norm_ext}'. "
                f"Supported: {sorted(list(cls._registry.keys()))}"
            )
        return parser_cls()
