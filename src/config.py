"""Central configuration for the agentic RAG prototype."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DOCS_DIR = Path(os.environ.get("DOCS_DIR", REPO_ROOT / "data" / "docs"))
TOC_DIR = Path(os.environ.get("TOC_DIR", REPO_ROOT / "data" / "toc"))
DOCS_MANIFEST_PATH = Path(
    os.environ.get("DOCS_MANIFEST_PATH", REPO_ROOT / "data" / "docs_manifest.json")
)
INDEX_DIR = Path(os.environ.get("INDEX_DIR", REPO_ROOT / "data" / "index"))

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2:3b")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1200"))  # characters, not words
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))  # characters, not words

RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "4"))
RETRIEVAL_RELEVANCE_THRESHOLD = float(
    os.environ.get("RETRIEVAL_RELEVANCE_THRESHOLD", "0.35")
)

LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))
