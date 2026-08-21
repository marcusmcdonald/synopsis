"""Base types and protocols for extraction adapters.

This module defines the core data structures and callable contracts that all
extractor adapters must implement.
"""

from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class ExtractionResult[ModelT]:
    """Framework-independent result returned by every extractor adapter.

    Attributes:
        extracted_text: The extracted document summary. For structured
            extraction this is the model serialized as JSON.
        structured_output: The validated Pydantic model, when one was supplied.
    """

    extracted_text: str
    structured_output: ModelT | None = None

    @property
    def output(self) -> str | ModelT:
        """Return the structured model when available, otherwise plain text."""

        if self.structured_output is not None:
            return self.structured_output
        return self.extracted_text


class MetadataExtractor[ModelT](Protocol):
    """Callable contract expected by summarize.py.

    All extractor adapters must implement this protocol to be compatible
    with the extraction pipeline.
    """

    def __call__(
        self,
        *,
        raw_text: str,
        char_budget: str | int,
    ) -> ExtractionResult[ModelT]:
        """Summarize a text document to fit within the character budget.

        Args:
            raw_text: The document text to summarize.
            char_budget: Maximum number of characters for the extracted output.

        Returns:
            ExtractionResult containing the document summary.
        """
        ...
