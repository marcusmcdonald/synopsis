import json

from pydantic import BaseModel, Field

from extractors.structured import json_schema, parse_output, schema_instruction


class SampleModel(BaseModel):
    title: str
    summary: str
    details: str = Field(default="N/A")


def test_parse_output_plain_text():
    raw = "Just plain text summary."
    serialized, structured = parse_output(raw, None)
    assert serialized == raw
    assert structured is None


def test_parse_output_clean_json():
    data = {"title": "Test Title", "summary": "Test Summary", "details": "Specifics"}
    raw = json.dumps(data)
    serialized, structured = parse_output(raw, SampleModel)
    assert structured is not None
    assert structured.title == "Test Title"
    assert structured.summary == "Test Summary"
    assert structured.details == "Specifics"
    assert json.loads(serialized)["title"] == "Test Title"


def test_parse_output_markdown_fence():
    raw = """
Here is your JSON output:
```json
{
  "title": "Fenced Title",
  "summary": "Fenced Summary",
  "details": "Fenced Details"
}
```
Hope this helps!
"""
    serialized, structured = parse_output(raw, SampleModel)
    assert structured is not None
    assert structured.title == "Fenced Title"
    assert structured.summary == "Fenced Summary"
    assert structured.details == "Fenced Details"
    assert "Fenced Title" in serialized


def test_parse_output_invalid_json():
    raw = "Invalid JSON that cannot be parsed into model"
    serialized, structured = parse_output(raw, SampleModel)
    assert structured is None
    assert serialized == raw


def test_json_schema():
    schema = json_schema(SampleModel)
    assert "properties" in schema
    assert "title" in schema["properties"]
    assert "summary" in schema["properties"]


def test_schema_instruction():
    instruction = schema_instruction(SampleModel)
    assert "JSON Schema" in instruction
    assert "title" in instruction

    none_instruction = schema_instruction(None)
    assert none_instruction == ""
