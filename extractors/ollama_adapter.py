"""Ollama direct client extractor adapter.

This module provides a direct Ollama Python client implementation compatible
with the extractor callable contract defined in base.py.
"""

from typing import Any, TypeVar

from pydantic import BaseModel

from .base import ExtractionResult
from .prompt import SYSTEM_PROMPT, build_user_prompt
from .structured import json_schema, parse_output

ModelT = TypeVar("ModelT", bound=BaseModel)


class OllamaExtractor[ModelT]:
    """Direct Ollama Python-client implementation."""

    def __init__(
        self,
        *,
        model: str = "qwen3:8b",
        host: str = "http://localhost:11434",
        temperature: float = 0.2,
        output_model: type[ModelT] | None = None,
    ) -> None:
        """Initialize the Ollama extractor.

        Args:
            model: The Ollama model name (default "qwen3:8b").
            host: Ollama server host URL (default "http://localhost:11434").
            temperature: Sampling temperature for generation (default 0.2).
            output_model: Optional Pydantic model defining the response shape.
        """
        # Import lazily so users of other adapters do not need ollama installed.
        import ollama

        self._client = ollama.Client(host=host)
        self._model = model
        self._temperature = temperature
        self._output_model = output_model

    def __call__(
        self,
        *,
        raw_text: str,
        char_budget: str | int,
    ) -> ExtractionResult[ModelT]:
        """Summarize a document using Ollama chat completion.

        Args:
            raw_text: The document text to summarize.
            char_budget: Maximum number of characters for the extracted output.

        Returns:
            ExtractionResult containing the document summary.
        """
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        raw_text, char_budget, self._output_model
                    ),
                },
            ],
            "options": {"temperature": self._temperature},
        }
        if self._output_model is not None:
            request["format"] = json_schema(self._output_model)
        response = self._client.chat(**request)

        # Support both dict-like and object-like versions of the Ollama client.
        if isinstance(response, dict):
            message = response["message"]
        else:
            message = response.message

        if isinstance(message, dict):
            text = message["content"]
        else:
            text = message.content

        serialized, structured = parse_output(text or "", self._output_model)
        return ExtractionResult(serialized, structured)
