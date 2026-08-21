SYSTEM_PROMPT = """\
You summarize source code and technical documents into dense technical summaries.

IMPORTANT:
- char_budget applies to the document summary content (excluding JSON syntax/keys if structured).
- Ensure all quotes, backslashes, and special characters inside JSON string values are properly escaped.

STRICT CONSTRAINTS:
1. DO NOT write complete conversational sentences or introductory prose.
2. Use telegraphic style. Example: Bridges field-bus to cloud. gRPC port 7443.
3. Retain ALL specific nouns, identifiers, port numbers, configuration keys, and file names.
4. Remove filler words such as: "This file is responsible for..." and "It also manages...".
5. Keep explanations dense, concise, and focused on technical specifics.
6. Separate multi-point details with semicolons (;).
""".strip()


from pydantic import BaseModel

from .structured import schema_instruction


def build_user_prompt(
    raw_text: str,
    char_budget: str | int,
    output_model: type[BaseModel] | None = None,
) -> str:
    """Build the per-request portion of the extraction prompt."""

    return f"""\
Summarize the document below.

ABSOLUTE MAXIMUM LENGTH: {char_budget} characters.
The output MUST contain no more than {char_budget} characters.

<document>
{raw_text}
</document>{schema_instruction(output_model)}
""".strip()


def build_prompt(
    raw_text: str,
    char_budget: str | int,
    output_model: type[BaseModel] | None = None,
) -> str:
    """Build a single prompt for frameworks that do not expose chat roles."""

    return (
        f"{SYSTEM_PROMPT}\n\n{build_user_prompt(raw_text, char_budget, output_model)}"
    )
