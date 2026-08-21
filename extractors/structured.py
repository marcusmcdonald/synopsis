"""Shared Pydantic structured-output support for extractor adapters."""

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def schema_instruction(output_model: type[BaseModel] | None) -> str:
    """Build an instruction containing the exact JSON Schema to return."""

    if output_model is None:
        return ""
    schema = json.dumps(json_schema(output_model), separators=(",", ":"))
    return (
        "\n\nReturn ONLY one JSON object that conforms exactly to this JSON Schema. "
        "Do not wrap it in Markdown or add commentary:\n" + schema
    )


def parse_output[ModelT](
    text: str, output_model: type[ModelT] | None
) -> tuple[str, ModelT | None]:
    """Validate a model response and retain a canonical JSON representation."""
    if output_model is None:
        return text, None

    candidate = text.strip()

    # 1. Check for markdown code blocks (e.g. ```json { ... } ```)
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
    else:
        # 2. Check for outer json object bounds if there is leading/trailing text
        outer_match = re.search(r"(\{.*\})", candidate, re.DOTALL)
        if outer_match:
            candidate = outer_match.group(1).strip()

    try:
        value = output_model.model_validate_json(candidate)
        serialized = value.model_dump_json()
        return serialized, value
    except Exception:  # noqa: BLE001
        # Fallback: try standard json decode then model_validate
        try:
            raw_dict = json.loads(candidate)
            value = output_model.model_validate(raw_dict)
            return value.model_dump_json(), value
        except Exception:  # noqa: BLE001
            return candidate, None


def json_schema(output_model: type[BaseModel]) -> dict[str, Any]:
    """Return a Pydantic model's JSON Schema."""
    return output_model.model_json_schema()
