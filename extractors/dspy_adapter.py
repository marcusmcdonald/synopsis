"""DSPy-based extractor adapter.

This module provides a DSPy implementation compatible with the extractor
callable contract defined in base.py.
"""

from typing import TypeVar

from pydantic import BaseModel

from .base import ExtractionResult
from .structured import parse_output, schema_instruction

ModelT = TypeVar("ModelT", bound=BaseModel)


class DSPyExtractor[ModelT]:
    """DSPy implementation compatible with the extractor callable contract.

    Uses DSPy's Predict module with a signature optimized for document summarization.
    """

    def __init__(
        self,
        *,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        output_model: type[ModelT] | None = None,
        **lm_kwargs,
    ) -> None:
        """Initialize the DSPy extractor.

        Args:
            model: The DSPy model identifier (e.g., "ollama_chat/qwen3:8b").
            api_base: Base URL for the API endpoint (optional).
            api_key: API key for authentication (optional).
            temperature: Sampling temperature for generation (default 0.2).
            output_model: Optional Pydantic model defining the response shape.
            **lm_kwargs: Additional keyword arguments passed to the DSPy LM.
        """
        # Import lazily so users of other adapters do not need DSPy installed.
        import dspy

        class SummarizeDocument(dspy.Signature):
            """Summarize the provided technical document or source code."""

            raw_text = dspy.InputField(desc="The document or code text to summarize")
            char_budget = dspy.InputField(
                desc="Target maximum character budget for the summary"
            )
            output_schema = dspy.InputField(
                desc="Optional JSON Schema instruction. When present, return a JSON object conforming to it."
            )
            extracted_text = dspy.OutputField(
                desc="Technical summary conforming to the requested schema or character budget."
            )

        lm_options = {
            "temperature": temperature,
            **lm_kwargs,
        }
        if api_base is not None:
            lm_options["api_base"] = api_base
        if api_key is not None:
            lm_options["api_key"] = api_key

        self._lm = dspy.LM(model, **lm_options)
        dspy.configure(lm=self._lm)

        self._predictor = dspy.Predict(SummarizeDocument)
        self._output_model = output_model

    def __call__(
        self,
        *,
        raw_text: str,
        char_budget: str | int,
    ) -> ExtractionResult[ModelT]:
        """Summarize a document using the DSPy predictor.

        Args:
            raw_text: The document text to summarize.
            char_budget: Maximum number of characters for the extracted output.

        Returns:
            ExtractionResult containing the document summary.
        """
        result = self._predictor(
            raw_text=raw_text,
            char_budget=str(char_budget),
            output_schema=schema_instruction(self._output_model),
        )
        extracted = getattr(result, "extracted_text", "") or ""
        serialized, structured = parse_output(extracted, self._output_model)
        return ExtractionResult(serialized, structured)
