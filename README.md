# Synopsis

Synopsis is a fast, LLM-powered codebase and document indexing engine. It scans codebases and documents, generates dense telegraphic technical summaries and normalized vector embeddings, stores them in SQLite (with optional YAML export), and provides an interactive Model Context Protocol (MCP) server for fast semantic discovery and micro-inspection.

## Features

- **Document Ingestion**: Supports raw source files, markdown, text, plus 50+ document formats via [Docling](https://github.com/DS4SD/docling) (PDF, DOCX, PPTX, XLSX, HTML, images with OCR, and more).
- **Flexible LLM Extractor Adapters**:
  - **OpenAI-compatible**: Native structured output parsing (`gpt-4o-mini`, vLLM, LM Studio, OpenRouter).
  - **Ollama**: Local LLM execution (`qwen`, `llama3`, `mistral`, etc.).
  - **DSPy**: Declarative programmatic prompt pipeline.
  - **Function**: Custom python callable adapter.
- **Vector Search & Embedding Storage**: Serialized float32 normalized embeddings stored directly in SQLite alongside document summaries and metadata.
- **FastMCP Intelligence Server**:
  - `get_repository_map`: Vector similarity search for discovering modules and repository layout biased toward any topic.
  - `query_document_corpus`: Safe, read-only SQL inspection engine (`PRAGMA query_only = ON;`).
  - `reload_index`: Live reload index without restarting the server.
- **Optimized Performance**: Top-down pruned directory traversal (skips `.git`, `node_modules`, `venv`, etc. before descending) and Write-Ahead Logging (WAL) SQLite transactions.

## Installation

Using `uv`:
```bash
uv sync
```

Or install in editable mode:
```bash
pip install -e .
```

## CLI Usage

### Build a Document Corpus

```bash
# Build index for the current directory using gpt-4o-mini
synopsis . --db corpus.db

# Use Ollama locally
synopsis ./my-project --db corpus.db --backend ollama --model qwen2.5-coder:7b

# Export to YAML as well
synopsis ./my-project --db corpus.db --yaml corpus.yaml

# Custom character budget & ignored directories
synopsis ./my-project --char-budget 1500 --ignore-dir extra_cache
```

### Synopsis CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `directory` | *Required* | Directory to traverse |
| `--db` | `corpus.db` | Path to destination SQLite database |
| `-y, --yaml` | `None` | Optional YAML export path |
| `-m, --model` | `gpt-4o-mini` | LLM extractor model name |
| `--embed-model` | `all-MiniLM-L6-v2` | SentenceTransformer embedding model |
| `-b, --backend` | `auto` | Backend adapter (`auto`, `openai`, `ollama`, `dspy`) |
| `--base-url` | `None` | Custom API base URL |
| `--api-key` | `None` | API key (defaults to environment `OPENAI_API_KEY`) |
| `--char-budget` | `2000` | Target character budget for summary |
| `--max-file-chars` | `120000` | Max characters read per file before truncation |
| `--ignore-dir` | `None` | Additional directory names to ignore |

## Running the MCP Server

Start the FastMCP server for AI assistants (e.g. Claude Desktop, Antigravity):

```bash
synopsis-server --db corpus.db
```

Or configure via environment variables:
```bash
export SYNOPSIS_DB_PATH="corpus.db"
export SYNOPSIS_EMBED_MODEL="all-MiniLM-L6-v2"
synopsis-server
```

## MCP Tools & Resources

- `get_repository_map(query, max_items=5)`: Semantic search over document corpus.
- `query_document_corpus(query)`: Executes read-only SQL queries against the `document_corpus` table.
- `reload_index()`: Reloads documents and embeddings from SQLite.
- `schema://document_corpus`: Returns the SQL schema of the corpus table.

## Running Tests

```bash
uv run pytest
```

## License

MIT
