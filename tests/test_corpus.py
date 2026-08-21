import json
import sqlite3
from pathlib import Path

import numpy as np

from build_corpus import (
    DocumentSummary,
    build_corpus,
    extract_content_from_file,
    init_sqlite_db,
    save_record_to_db,
)
from extractors.function_adapter import FunctionExtractor


def test_sqlite_init_and_save(tmp_path: Path):
    db_path = tmp_path / "test_corpus.db"
    conn = sqlite3.connect(db_path)
    init_sqlite_db(conn)

    vec = np.zeros(384, dtype=np.float32).tobytes()
    save_record_to_db(
        conn=conn,
        doc_id="main.py",
        title="Main Module",
        summary="Entry point of application.",
        details="Runs server on port 8000.",
        content_embedding=vec,
    )
    conn.commit()

    cursor = conn.cursor()
    cursor.execute(
        "SELECT doc_id, title, summary, details, content_embedding FROM document_corpus"
    )
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "main.py"
    assert row[1] == "Main Module"
    assert row[2] == "Entry point of application."
    assert row[3] == "Runs server on port 8000."
    assert len(row[4]) == 384 * 4
    conn.close()


def test_extract_content_truncation(tmp_path: Path):
    large_file = tmp_path / "large.py"
    large_file.write_text("A" * 500, encoding="utf-8")

    content, method = extract_content_from_file(large_file, max_chars=100)
    assert content is not None
    assert method in ("code_or_text", "docling")
    assert len(content) < 500
    assert "[... content truncated ...]" in content


def test_build_corpus_end_to_end(tmp_path: Path):
    # Setup test workspace
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "app.py").write_text("def run(): pass", encoding="utf-8")
    (src_dir / "utils.py").write_text("def helper(): pass", encoding="utf-8")

    # Ignored directory
    git_dir = src_dir / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git internal", encoding="utf-8")

    db_path = tmp_path / "output.db"
    yaml_path = tmp_path / "output.yaml"

    def mock_generate(prompt: str) -> str:
        return json.dumps(
            {
                "title": "Mock Title",
                "summary": "Mock summary overview.",
                "details": "Mock details specifics.",
            }
        )

    mock_extractor = FunctionExtractor(
        generate=mock_generate, output_model=DocumentSummary
    )

    corpus = build_corpus(
        root_dir=src_dir,
        db_path=db_path,
        yaml_output=yaml_path,
        extractor=mock_extractor,
    )

    assert "app.py" in corpus
    assert "utils.py" in corpus
    assert ".git/config" not in corpus
    assert db_path.exists()
    assert yaml_path.exists()
