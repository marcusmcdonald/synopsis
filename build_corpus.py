import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from binaryornot.check import is_binary
from constants import DEFAULT_IGNORED_DIRS, DOCLING_DOCUMENT_EXTENSIONS
from docling.datamodel.base_models import FormatToExtensions
from docling.document_converter import ConversionStatus, DocumentConverter
from extractors import (
    DSPyExtractor,
    MetadataExtractor,
    OllamaExtractor,
    OpenAIExtractor,
)
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Type alias for document records in memory
type DocumentSummaries = dict[str, dict[str, Any]]


class DocumentSummary(BaseModel):
    title: str = Field(description="A concise, descriptive title.")
    summary: str = Field(description="A high-level 1-2 sentence overview.")
    details: str = Field(description="Key technical specifics.")


def init_sqlite_db(db_path: Path) -> None:
    """Initializes SQLite database and creates table with content_embedding column."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_corpus (
                doc_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                details TEXT NOT NULL,
                content_embedding BLOB
            );
            """
        )
        conn.commit()


def save_record_to_db(
    db_path: Path,
    doc_id: str,
    title: str,
    summary: str,
    details: str,
    content_embedding: bytes | None = None,
) -> None:
    """Inserts or updates a single document record including serialized embedding bytes."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO document_corpus (doc_id, title, summary, details, content_embedding)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                title = excluded.title,
                summary = excluded.summary,
                details = excluded.details,
                content_embedding = excluded.content_embedding;
            """,
            (doc_id, title, summary, details, content_embedding),
        )
        conn.commit()


def get_all_docling_extensions(converter: DocumentConverter | None = None) -> set[str]:
    """Returns all file extensions supported by the DocumentConverter."""
    conv = converter or DocumentConverter()
    extensions: set[str] = set()
    for fmt in conv.allowed_formats:
        for ext in FormatToExtensions.get(fmt, []):
            extensions.add(f".{ext.lower()}")
    return extensions | DOCLING_DOCUMENT_EXTENSIONS


def extract_content_from_file(
    file_path: Path,
    converter: DocumentConverter | None = None,
    docling_extensions: set[str] | None = None,
) -> tuple[str | None, str]:
    """Extracts text content from a file using Docling or direct text decoding."""
    exts = (
        docling_extensions
        if docling_extensions is not None
        else get_all_docling_extensions(converter)
    )
    file_ext = file_path.suffix.lower()

    if file_ext in exts:
        conv = converter or DocumentConverter()
        try:
            res = conv.convert(file_path, raises_on_error=False)
            if res.status in (
                ConversionStatus.SUCCESS,
                ConversionStatus.PARTIAL_SUCCESS,
            ):
                text = res.document.export_to_markdown()
                if text and text.strip():
                    return text.strip(), "docling"
                return None, "empty_docling_output"
        except (OSError, ValueError, RuntimeError) as e:
            if not is_binary(str(file_path)):
                try:
                    fallback = file_path.read_text(encoding="utf-8", errors="replace")
                    if fallback.strip():
                        return fallback, "text_fallback"
                except OSError, UnicodeDecodeError:
                    return None, f"fallback_read_error: {e}"
            return None, f"docling_error: {e}"

    try:
        if is_binary(str(file_path)):
            conv = converter or DocumentConverter()
            try:
                res = conv.convert(file_path, raises_on_error=False)
                if res.status in (
                    ConversionStatus.SUCCESS,
                    ConversionStatus.PARTIAL_SUCCESS,
                ):
                    text = res.document.export_to_markdown()
                    if text and text.strip():
                        return text.strip(), "docling"
            except OSError, ValueError, RuntimeError:
                return None, "unsupported_binary"
            return None, "unsupported_binary"
    except (OSError, ValueError) as e:
        return None, f"binary_check_error: {e}"

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            return None, "empty_text_file"
        return content, "code_or_text"
    except (OSError, UnicodeDecodeError) as e:
        return None, f"read_error: {e}"


def create_extractor(
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    api_key: str = "placeholder",
    output_model: type[BaseModel] = DocumentSummary,
    backend: str = "auto",
) -> MetadataExtractor[Any]:
    """Instantiates an appropriate extractor adapter."""
    normalized_backend = backend.lower()

    if normalized_backend == "dspy" or (
        normalized_backend == "auto" and model.startswith("dspy:")
    ):
        dspy_model = model.removeprefix("dspy:")
        return DSPyExtractor(
            model=dspy_model,
            api_base=base_url,
            api_key=api_key
            if api_key not in ("placeholder", "dummy", "unused")
            else None,
            output_model=output_model,
        )

    if normalized_backend == "ollama":
        model_name = model.split("/", 1)[1] if model.startswith("ollama/") else model
        return OllamaExtractor(
            model=model_name,
            host=base_url or "http://localhost:11434",
            output_model=output_model,
        )

    if normalized_backend == "openai":
        return OpenAIExtractor(
            model=model,
            base_url=base_url,
            api_key=api_key if api_key != "placeholder" else "unused",
            output_model=output_model,
        )

    if (base_url and "11434" in base_url) or any(
        model.lower().startswith(p)
        for p in ("qwen", "llama", "mistral", "phi", "gemma", "ollama")
    ):
        model_name = model.split("/", 1)[1] if model.startswith("ollama/") else model
        return OllamaExtractor(
            model=model_name,
            host=base_url or "http://localhost:11434",
            output_model=output_model,
        )

    return OpenAIExtractor(
        model=model,
        base_url=base_url,
        api_key=api_key if api_key != "placeholder" else "unused",
        output_model=output_model,
    )


def build_corpus(
    root_dir: Path,
    db_path: Path | None = Path("corpus.db"),
    yaml_output: Path | None = None,
    model: str = "gpt-4o-mini",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    base_url: str | None = None,
    api_key: str = "placeholder",
    backend: str = "auto",
    extractor: MetadataExtractor[DocumentSummary] | None = None,
    converter: DocumentConverter | None = None,
    char_budget: int = 2_000,
    ignored_dirs: set[str] | list[str] | None = None,
) -> DocumentSummaries:
    """Traverses root_dir, generates summaries/embeddings, writes to SQLite DB."""
    if extractor is None:
        extractor = create_extractor(
            model=model,
            base_url=base_url,
            api_key=api_key,
            output_model=DocumentSummary,
            backend=backend,
        )

    if converter is None:
        converter = DocumentConverter()

    print(f"Loading embedding model: {embedding_model_name}")
    embedder = SentenceTransformer(embedding_model_name)

    if db_path is not None:
        init_sqlite_db(db_path)

    ignore_set = set(ignored_dirs) if ignored_dirs is not None else DEFAULT_IGNORED_DIRS
    docling_exts = get_all_docling_extensions(converter)

    corpus: DocumentSummaries = {}
    root_path = root_dir.resolve()
    resolved_db = db_path.resolve() if db_path is not None else None
    resolved_yaml = yaml_output.resolve() if yaml_output is not None else None

    print(f"Scanning directory: {root_path}")

    for file_path in sorted(root_path.rglob("*")):
        if not file_path.is_file():
            continue

        resolved_file = file_path.resolve()
        if (
            resolved_file == resolved_db
            or (resolved_yaml and resolved_file == resolved_yaml)
            or file_path.name.endswith("-journal")
            or file_path.name.endswith("-wal")
        ):
            continue

        rel_path = file_path.relative_to(root_path)

        if any(part in ignore_set for part in rel_path.parts[:-1]):
            continue

        doc_key = rel_path.as_posix()

        content, method = extract_content_from_file(
            file_path,
            converter=converter,
            docling_extensions=docling_exts,
        )

        if content is None:
            continue

        print(f"[+] Summarizing & Embedding ({method}): {rel_path} -> {doc_key}")
        try:
            summary = extractor(raw_text=content, char_budget=char_budget)

            if summary.structured_output is not None:
                record = summary.structured_output.model_dump()
            else:
                record = {
                    "title": file_path.stem.replace("_", " ").title(),
                    "summary": summary.extracted_text,
                    "details": "Extracted via unstructured fallback.",
                }

            text_to_embed = (
                f"{record['title']}: {record['summary']}\n{record['details']}"
            )
            embedding_vec = embedder.encode(text_to_embed, normalize_embeddings=True)
            embedding_bytes = embedding_vec.astype(np.float32).tobytes()

            corpus[doc_key] = record

            if db_path is not None:
                save_record_to_db(
                    db_path=db_path,
                    doc_id=doc_key,
                    title=record["title"],
                    summary=record["summary"],
                    details=record["details"],
                    content_embedding=embedding_bytes,
                )

        except Exception as e:  # noqa: BLE001
            print(f"[!] Error processing {rel_path}: {e}")

    if yaml_output is not None:
        yaml_output.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_output, "w", encoding="utf-8") as f:
            yaml.dump(
                corpus,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        print(f"[+] Generated YAML export at: {yaml_output}")

    if db_path is not None:
        print(f"\nSuccessfully stored {len(corpus)} records in {db_path}")

    return corpus
