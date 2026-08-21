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
    # Some compatible/local models still add a fence despite explicit prompting.
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    if hasattr(output_model, "model_validate_json"):
        value = output_model.model_validate_json(candidate)
        serialized = value.model_dump_json()
    else:  # Pydantic v1
        value = output_model.parse_raw(candidate)
        serialized = value.json()
    return serialized, value


def json_schema(output_model: type[BaseModel]) -> dict[str, Any]:
    """Return a Pydantic model's JSON Schema with a useful native format name."""

    if hasattr(output_model, "model_json_schema"):
        return output_model.model_json_schema()
    return output_model.schema()  # Pydantic v1
