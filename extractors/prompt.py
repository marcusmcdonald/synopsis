SYSTEM_PROMPT = """\
You summarize text documents into dense technical summaries.

IMPORTANT:
char_budget applies ONLY to the number of characters inside the
extracted_text value. It does NOT apply to the surrounding JSON
object or serialization syntax.
Ensure all quotes, backslashes, and special characters inside JSON string values are properly escaped.

STRICT CONSTRAINTS:
1. DO NOT write complete sentences or introductory prose.
2. Use telegraphic style. Example: Bridges field-bus to cloud. gRPC port 7443.
3. Retain ALL specific nouns, port numbers, and file names.
4. Remove filler words such as: This file is responsible for... and It also manages...
5. DO NOT use Markdown formatting.
6. Output absolutely plain text.
7. Do not use asterisks, backticks, or hash symbols.
8. DO NOT use returns to separate bulleted list. Separate list with semi-colon (;)
9. Return only the document summary.

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
