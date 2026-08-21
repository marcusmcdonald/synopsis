import sqlite3
from pathlib import Path

import numpy as np

import server
from build_corpus import init_sqlite_db, save_record_to_db
from server import CorpusIndex, get_repository_map, query_document_corpus, reload_index


def test_server_index_and_queries(tmp_path: Path):
    db_path = tmp_path / "test_server.db"
    conn = sqlite3.connect(db_path)
    init_sqlite_db(conn)

    # Insert 2 test documents with dummy embeddings
    vec1 = np.ones(384, dtype=np.float32)
    vec1 = (vec1 / np.linalg.norm(vec1)).astype(np.float32).tobytes()

    vec2 = -np.ones(384, dtype=np.float32)
    vec2 = (vec2 / np.linalg.norm(vec2)).astype(np.float32).tobytes()

    save_record_to_db(
        conn=conn,
        doc_id="api/auth.py",
        title="Authentication API",
        summary="Handles user auth and JWT validation.",
        details="Endpoint POST /login, Port 8080",
        content_embedding=vec1,
    )
    save_record_to_db(
        conn=conn,
        doc_id="db/models.py",
        title="Database Models",
        summary="SQLAlchemy database schemas.",
        details="User and Session tables",
        content_embedding=vec2,
    )
    conn.commit()
    conn.close()

    # Instantiate CorpusIndex
    idx = CorpusIndex(db_path, "all-MiniLM-L6-v2")
    assert len(idx.doc_ids) == 2
    assert idx.dimension == 384

    # Test rank
    results = idx.rank("Authentication login endpoint", top_k=2)
    assert len(results) == 2
    assert "auth.py" in results[0]["doc_id"]

    # Point global server index to test DB
    server.INDEX = idx

    # Test get_repository_map tool
    repo_map = get_repository_map("Authentication")
    assert "Authentication API" in repo_map
    assert "api/auth.py" in repo_map

    # Test query_document_corpus tool
    sql_res = query_document_corpus(
        "SELECT doc_id, title FROM document_corpus WHERE doc_id = 'api/auth.py'"
    )
    assert "Authentication API" in sql_res
    assert "api/auth.py" in sql_res

    # Test query_document_corpus non-SELECT query rejection
    rejected = query_document_corpus("DELETE FROM document_corpus")
    assert "Error: Only SELECT queries are permitted" in rejected

    # Test reload_index tool
    reload_msg = reload_index()
    assert "Successfully reloaded index with 2 documents" in reload_msg
