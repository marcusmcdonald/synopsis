"""Extractor adapters for various LLM backends.

This package provides a unified interface for summarizing text documents using
different LLM providers (DSPy, Ollama, OpenAI, or custom functions).
"""

from .base import ExtractionResult, MetadataExtractor
from .dspy_adapter import DSPyExtractor
from .function_adapter import FunctionExtractor
from .ollama_adapter import OllamaExtractor
from .openai_adapter import OpenAIExtractor

__all__ = [
    "DSPyExtractor",
    "ExtractionResult",
    "FunctionExtractor",
    "MetadataExtractor",
    "OllamaExtractor",
    "OpenAIExtractor",
]
