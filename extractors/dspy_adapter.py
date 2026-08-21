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
            """
            Summarize the provided text document to fit strictly within the target character budget.

            IMPORTANT:
            char_budget applies ONLY to the number of characters inside the
            extracted_text value. It does NOT apply to the surrounding JSON
            object or serialization syntax.
            Ensure all quotes, backslashes, and special characters inside JSON string values are properly escaped.

            STRICT CONSTRAINTS:
            1. DO NOT write complete sentences or introductory prose.
            2. Use telegraphic style (e.g., "Bridges field-bus to cloud. gRPC port 7443.").
            3. Retain ALL specific nouns, port numbers, and file names.
            4. Remove all filler words (e.g., "This file is responsible for...", "It also manages...").
            5. DO NOT use any Markdown formatting in the extracted_text: value. Output the value to extracted_text as absolutely plain text. Do not use asterisks (**), backticks (`), or hash symbols (#).
            6. DO NOT use returns to separate bulleted list. Separate list with semi-colon (;)
            """

            raw_text = dspy.InputField(desc="The document text to summarize")
            char_budget = dspy.InputField(
                desc=(
                    "Maximum length of the extracted_text VALUE only. "
                    "Does not include JSON syntax, field names, quotes, or other "
                    "serialization overhead."
                )
            )
            output_schema = dspy.InputField(
                desc=(
                    "Optional JSON Schema instruction. When present, return "
                    "only a JSON object conforming to it."
                )
            )
            extracted_text = dspy.OutputField(
                desc=(
                    "Document summary. The VALUE of this field must contain "
                    "at most char_budget characters. Always return this field; "
                    "never return an error object."
                )
            )

        lm_options = {
            "temperature": temperature,
            **lm_kwargs,
        }
        if api_base is not None:
            lm_options["api_base"] = api_base
        if api_key is not None:
            lm_options["api_key"] = api_key

        lm = dspy.LM(model, **lm_options)
        dspy.configure(lm=lm)

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
        serialized, structured = parse_output(result.extracted_text, self._output_model)
        return ExtractionResult(serialized, structured)
