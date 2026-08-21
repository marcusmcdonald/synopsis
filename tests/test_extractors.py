import json

from pydantic import BaseModel

from build_corpus import DocumentSummary, create_extractor
from extractors.base import ExtractionResult
from extractors.function_adapter import FunctionExtractor
from extractors.ollama_adapter import OllamaExtractor
from extractors.openai_adapter import OpenAIExtractor


class DummyModel(BaseModel):
    title: str
    summary: str
    details: str


def test_function_extractor_unstructured():
    def mock_generate(prompt: str) -> str:
        return "Telegraphic summary of code."

    extractor = FunctionExtractor(generate=mock_generate, output_model=None)
    res: ExtractionResult = extractor(raw_text="print('hello')", char_budget=500)
    assert res.extracted_text == "Telegraphic summary of code."
    assert res.structured_output is None
    assert res.output == "Telegraphic summary of code."


def test_function_extractor_structured():
    def mock_generate(prompt: str) -> str:
        return json.dumps(
            {
                "title": "Module Init",
                "summary": "Initializes module.",
                "details": "Port 8080",
            }
        )

    extractor = FunctionExtractor(generate=mock_generate, output_model=DummyModel)
    res = extractor(raw_text="print('hello')", char_budget=500)
    assert res.structured_output is not None
    assert res.structured_output.title == "Module Init"
    assert res.output == res.structured_output


def test_create_extractor_factory():
    # OpenAI Backend
    openai_ext = create_extractor(
        model="gpt-4o-mini",
        backend="openai",
        api_key=None,
        output_model=DocumentSummary,
    )
    assert isinstance(openai_ext, OpenAIExtractor)
    assert openai_ext._model == "gpt-4o-mini"

    # Ollama Backend
    ollama_ext = create_extractor(
        model="llama3:8b",
        backend="ollama",
        output_model=DocumentSummary,
    )
    assert isinstance(ollama_ext, OllamaExtractor)
    assert ollama_ext._model == "llama3:8b"

    # Auto Backend resolution for qwen
    auto_ollama = create_extractor(
        model="qwen2.5:7b",
        backend="auto",
        output_model=DocumentSummary,
    )
    assert isinstance(auto_ollama, OllamaExtractor)
