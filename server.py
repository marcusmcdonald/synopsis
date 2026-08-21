import json
from pathlib import Path
import sqlite3
from typing import Any

from mcp.server.fastmcp import FastMCP
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize FastMCP Server
mcp = FastMCP("Corpus Intelligence Server")
DB_PATH = Path("corpus.db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class CorpusIndex:
    """Manages in-memory document state and vector similarity index."""

    def __init__(self, db_path: Path, model_name: str) -> None:
        self.db_path = db_path
        self.embedder = SentenceTransformer(model_name)
        self.doc_ids: list[str] = []
        self.titles: list[str] = []
        self.summaries: list[str] = []
        self.embeddings: np.ndarray = np.empty((0, 384), dtype=np.float32)
        self.load_index()

    def load_index(self) -> None:
        """Loads records and embeddings into memory for fast similarity ranking."""
        if not self.db_path.exists():
            return

        doc_ids, titles, summaries, raw_vecs = [], [], [], []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT doc_id, title, summary, content_embedding "
                "FROM document_corpus"
            )
            for doc_id, title, summary, blob in cursor.fetchall():
                doc_ids.append(doc_id)
                titles.append(title)
                summaries.append(summary)
                if blob is not None:
                    raw_vecs.append(np.frombuffer(blob, dtype=np.float32))
                else:
                    raw_vecs.append(np.zeros(384, dtype=np.float32))

        self.doc_ids = doc_ids
        self.titles = titles
        self.summaries = summaries
        if raw_vecs:
            self.embeddings = np.vstack(raw_vecs)

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
INDEX = CorpusIndex(DB_PATH, EMBEDDING_MODEL_NAME)


# MCP Resource: Database Schema Context
@mcp.resource("schema://document_corpus")
def get_schema() -> str:
    """Provides the exact SQL schema of the document corpus table."""
    return """
    CREATE TABLE document_corpus (
        doc_id TEXT PRIMARY KEY,       -- Relative file path / document key
        title TEXT NOT NULL,           -- Descriptive title
        summary TEXT NOT NULL,         -- 1-2 sentence high-level overview
        details TEXT NOT NULL,         -- Technical specifics & parameters
        content_embedding BLOB         -- 384-dim float32 embedding vector
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

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(sanitized)
            rows = cursor.fetchall()
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


if __name__ == "__main__":
    mcp.run()