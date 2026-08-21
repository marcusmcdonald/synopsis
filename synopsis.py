import argparse
from pathlib import Path

from build_corpus import build_corpus
from constants import DEFAULT_IGNORED_DIRS


def synopsis():
    parser = argparse.ArgumentParser(
        description="Generate a SQLite document corpus with vector embeddings."
    )
    parser.add_argument("directory", type=Path, help="Directory to traverse")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("corpus.db"),
        help="Path to SQLite DB",
    )
    parser.add_argument(
        "-y",
        "--yaml",
        type=Path,
        default=None,
        help="Optional YAML export path",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM extractor model",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "dspy", "ollama", "openai"],
        help="LLM extractor backend",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom API endpoint",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="dummy",
        help="API Key",
    )
    parser.add_argument(
        "--char-budget",
        type=int,
        default=2000,
        help="Target character budget",
    )
    parser.add_argument(
        "--ignore-dir",
        action="append",
        dest="ignored_dirs",
        help="Additional directories to ignore",
    )

    args = parser.parse_args()

    custom_ignored = None
    if args.ignored_dirs:
        custom_ignored = DEFAULT_IGNORED_DIRS | set(args.ignored_dirs)

    build_corpus(
        root_dir=args.directory,
        db_path=args.db,
        yaml_output=args.yaml,
        model=args.model,
        embedding_model_name=args.embed_model,
        base_url=args.base_url,
        api_key=args.api_key,
        backend=args.backend,
        char_budget=args.char_budget,
        ignored_dirs=custom_ignored,
    )


if __name__ == "__main__":
    synopsis()
