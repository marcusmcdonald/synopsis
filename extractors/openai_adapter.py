"""OpenAI-compatible API extractor adapter.

This module provides an adapter for OpenAI-compatible chat-completions APIs,
compatible with the extractor callable contract defined in base.py.

Works with providers such as OpenAI, OpenRouter, vLLM, LM Studio, and
other servers exposing an OpenAI-compatible /v1 endpoint.
"""

from typing import TypeVar

from pydantic import BaseModel

from .base import ExtractionResult
from .prompt import SYSTEM_PROMPT, build_user_prompt
from .structured import json_schema, parse_output

ModelT = TypeVar("ModelT", bound=BaseModel)


class OpenAIExtractor[ModelT]:
    """Adapter for OpenAI-compatible chat-completions APIs.

    Works with providers such as OpenAI, OpenRouter, vLLM, LM Studio, and
    other servers exposing an OpenAI-compatible /v1 endpoint.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        output_model: type[ModelT] | None = None,
        **client_kwargs,
    ) -> None:
        """Initialize the OpenAI extractor.

        Args:
            model: The model identifier to use for extraction.
            base_url: Base URL for the OpenAI-compatible API (optional).
            api_key: API key for authentication (optional, defaults to environment).
            temperature: Sampling temperature for generation (default 0.2).
            output_model: Optional Pydantic model defining the response shape.
            **client_kwargs: Additional keyword arguments passed to OpenAI client.
        """
        import os

        from openai import OpenAI

        effective_key = api_key or os.environ.get("OPENAI_API_KEY", "dummy")
        kwargs = dict(client_kwargs)
        kwargs["api_key"] = effective_key
        if base_url is not None:
            kwargs["base_url"] = base_url

        self._client = OpenAI(**kwargs)
        self._model = model
        self._temperature = temperature
        self._output_model = output_model

    def __call__(
        self,
        *,
        raw_text: str,
        char_budget: str | int,
    ) -> ExtractionResult[ModelT]:
        """Summarize a document using an OpenAI-compatible chat completion.

        Args:
            raw_text: The document text to summarize.
            char_budget: Maximum number of characters for the extracted output.

        Returns:
            ExtractionResult containing the document summary.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(raw_text, char_budget, self._output_model),
            },
        ]

        if self._output_model is not None:
            try:
                # Use native OpenAI structured outputs parsing
                response = self._client.beta.chat.completions.parse(
                    model=self._model,
                    temperature=self._temperature,
                    messages=messages,
                    response_format=self._output_model,
                )
                if response.choices:
                    structured = response.choices[0].message.parsed
                    if structured is not None:
                        return ExtractionResult(
                            extracted_text=structured.model_dump_json(),
                            structured_output=structured,
                        )
            except Exception:  # noqa: BLE001, S110
                # Fallback to standard chat completions if beta parse is unsupported (e.g. local vLLM/LM Studio)
                pass

        request = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": messages,
        }
        if self._output_model is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self._output_model.__name__,
                    "strict": True,
                    "schema": json_schema(self._output_model),
                },
            }

        response = self._client.chat.completions.create(**request)
        text = response.choices[0].message.content or "" if response.choices else ""
        serialized, structured = parse_output(text, self._output_model)
        return ExtractionResult(serialized, structured)
