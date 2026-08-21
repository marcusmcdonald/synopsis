import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

# Initialize FastMCP Server
mcp = FastMCP("Corpus Intelligence Server")

DEFAULT_DB_PATH = Path(os.environ.get("SYNOPSIS_DB_PATH", "corpus.db"))
DEFAULT_EMBEDDING_MODEL = os.environ.get("SYNOPSIS_EMBED_MODEL", "all-MiniLM-L6-v2")


class CorpusIndex:
    """Manages in-memory document state and vector similarity index."""

    def __init__(self, db_path: Path, model_name: str) -> None:
        self.db_path = db_path
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        if hasattr(self.embedder, "get_embedding_dimension"):
            self.dimension = int(self.embedder.get_embedding_dimension() or 384)
        else:
            self.dimension = int(
                self.embedder.get_sentence_embedding_dimension() or 384
            )
        self.doc_ids: list[str] = []
        self.titles: list[str] = []
        self.summaries: list[str] = []
        self.embeddings: np.ndarray = np.empty((0, self.dimension), dtype=np.float32)
        self.load_index()

    def load_index(self) -> int:
        """Loads records and embeddings into memory for fast similarity ranking."""
        self.doc_ids = []
        self.titles = []
        self.summaries = []
        self.embeddings = np.empty((0, self.dimension), dtype=np.float32)

        if not self.db_path.exists():
            return 0

        doc_ids, titles, summaries, raw_vecs = [], [], [], []

        db_uri = f"file:{self.db_path.resolve()}?mode=ro"
        with sqlite3.connect(db_uri, uri=True) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT doc_id, title, summary, content_embedding FROM document_corpus"
            )
            for doc_id, title, summary, blob in cursor.fetchall():
                doc_ids.append(doc_id)
                titles.append(title)
                summaries.append(summary)
                if blob is not None:
                    vec = np.frombuffer(blob, dtype=np.float32)
                    if vec.shape[0] == self.dimension:
                        raw_vecs.append(vec)
                    else:
                        raw_vecs.append(np.zeros(self.dimension, dtype=np.float32))
                else:
                    raw_vecs.append(np.zeros(self.dimension, dtype=np.float32))

        self.doc_ids = doc_ids
        self.titles = titles
        self.summaries = summaries
        if raw_vecs:
            self.embeddings = np.vstack(raw_vecs)

        return len(self.doc_ids)

    def rank(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Computes cosine similarity ranking against the query embedding."""
        if len(self.doc_ids) == 0 or self.embeddings.size == 0:
            return []

        query_vec = self.embedder.encode(query, normalize_embeddings=True)
        # Cosine similarity for normalized vectors is simply the dot product
        scores = np.dot(self.embeddings, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "doc_id": self.doc_ids[idx],
                "title": self.titles[idx],
                "summary": self.summaries[idx],
                "score": float(scores[idx]),
            }
            for idx in top_indices
        ]


# Initialize index globally on server launch
INDEX = CorpusIndex(DEFAULT_DB_PATH, DEFAULT_EMBEDDING_MODEL)


# MCP Resource: Database Schema Context
@mcp.resource("schema://document_corpus")
def get_schema() -> str:
    """Provides the exact SQL schema of the document corpus table."""
    return f"""
    CREATE TABLE document_corpus (
        doc_id TEXT PRIMARY KEY,       -- Relative file path / document key
        title TEXT NOT NULL,           -- Descriptive title
        summary TEXT NOT NULL,         -- 1-2 sentence high-level overview
        details TEXT NOT NULL,         -- Technical specifics & parameters
        content_embedding BLOB         -- {INDEX.dimension}-dim float32 embedding vector
    );
    """


# MCP Tool 1: Macro Orientation (Repository Map)
@mcp.tool()
def get_repository_map(query: str, max_items: int = 5) -> str:
    """Provides a high-level structural map of the repository/documents biased

    toward a specific topic using vector similarity ranking. Always run this
    first to discover relevant doc_ids, modules, and architecture layout.
    """
    results = INDEX.rank(query=query, top_k=max_items)
    if not results:
        return f"No index entries available for query: '{query}'"

    lines = [f"**Document Context Map for '{query}':**\n"]
    for idx, item in enumerate(results):
        prefix = "Primary Focus" if idx == 0 else "Related Context"
        lines.append(
            f"- **[{prefix}] {item['title']}** (`{item['doc_id']}`, Score: {item['score']:.3f})"
        )
        lines.append(f"  - *Summary:* {item['summary']}")

    return "\n".join(lines)


# MCP Tool 2: Micro Inspection (SQL Engine)
@mcp.tool()
def query_document_corpus(query: str) -> str:
    """Executes a read-only SQLite query against the `document_corpus` table.

    Use this to inspect exact technical implementation details, parameters,
    or run keyword filters for doc_ids discovered in the repository map.
    """
    sanitized = query.strip()
    if not sanitized.upper().startswith("SELECT"):
        return "Error: Only SELECT queries are permitted on this corpus."

    if not INDEX.db_path.exists():
        return f"Error: Database file does not exist at {INDEX.db_path}"

    try:
        db_uri = f"file:{INDEX.db_path.resolve()}?mode=ro"
        with sqlite3.connect(db_uri, uri=True) as conn:
            conn.execute("PRAGMA query_only = ON;")
            cursor = conn.cursor()
            cursor.execute(sanitized)
            rows = cursor.fetchall()
            if cursor.description is None:
                return "[]"

            columns = [col[0] for col in cursor.description]

            formatted = []
            for row in rows:
                row_dict = {}
                for col, val in zip(columns, row):
                    if col == "content_embedding" and val is not None:
                        row_dict[col] = f"<BLOB {len(val)} bytes>"
                    else:
                        row_dict[col] = val
                formatted.append(row_dict)

            return json.dumps(formatted, indent=2)

    except Exception as e:  # noqa: BLE001
        return f"SQL Execution Error: {e}"


# MCP Tool 3: Index Reload
@mcp.tool()
def reload_index() -> str:
    """Reloads the in-memory document corpus and embeddings from the SQLite database."""
    count = INDEX.load_index()
    return f"Successfully reloaded index with {count} documents from {INDEX.db_path}."


def main() -> None:
    """CLI entry point for launching the MCP server."""
    parser = argparse.ArgumentParser(description="Start the Synopsis MCP Server.")
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model name",
    )
    args = parser.parse_args()

    global INDEX
    if args.db != DEFAULT_DB_PATH or args.embed_model != DEFAULT_EMBEDDING_MODEL:
        INDEX = CorpusIndex(args.db, args.embed_model)

    mcp.run()


if __name__ == "__main__":
    main()
