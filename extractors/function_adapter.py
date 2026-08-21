from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

from .base import ExtractionResult
from .prompt import build_prompt
from .structured import parse_output

ModelT = TypeVar("ModelT", bound=BaseModel)


class FunctionExtractor[ModelT]:
    """
    Adapter for any framework that can be represented as:

        generate(prompt: str) -> str

    When ``output_model`` is set, its JSON Schema is included in the prompt and
    the returned JSON is validated into that Pydantic model.
    """

    def __init__(
        self,
        generate: Callable[[str], str],
        *,
        output_model: type[ModelT] | None = None,
    ) -> None:
        self._generate = generate
        self._output_model = output_model

    def __call__(
        self,
        *,
        raw_text: str,
        char_budget: str | int,
    ) -> ExtractionResult[ModelT]:
        """Summarize the document supplied in ``raw_text``."""

        prompt = build_prompt(raw_text, char_budget, self._output_model)
        text = self._generate(prompt)
        serialized, structured = parse_output(text or "", self._output_model)
        return ExtractionResult(serialized, structured)
