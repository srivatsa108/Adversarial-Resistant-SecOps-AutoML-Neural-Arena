"""Parser factory for resolving file parsers based on format."""

from typing import Dict, Type
from src.interfaces.parser import BaseFileParser
from src.parsers.csv_parser import CSVFileParser
from src.parsers.json_parser import JSONFileParser
from src.parsers.pcap_parser import PCAPFileParser
from src.parsers.pdf_parser import PDFFileParser


class ParserFactory:
    """Factory resolving appropriate parser strategies for supported file extensions."""

    _registry: Dict[str, Type[BaseFileParser]] = {
        "csv": CSVFileParser,
        "json": JSONFileParser,
        "pdf": PDFFileParser,
        "pcap": PCAPFileParser,
        "pcapng": PCAPFileParser,
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
        ext = extension.lower().strip().lstrip(".")
        if ext not in cls._registry:
            supported = ", ".join(cls._registry.keys())
            raise ValueError(f"No parser registered for '.{ext}'. Supported: {supported}")

        return cls._registry[ext]()
