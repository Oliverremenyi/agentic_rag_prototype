"""FAISS-backed similarity search over ingested document chunks.

Builds/persists a small local vector index (no external vector DB service).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src import config
from src.llm.local_llm import get_embeddings
from src.rag.ingest import Chunk, build_chunks, load_reports

_METADATA_FILENAME = "chunks.json"
_INDEX_FILENAME = "index.faiss"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def build_index(docs_dir: Path | None = None, index_dir: Path | None = None) -> int:
    """Ingest documents, embed chunks, and persist a FAISS index. Returns chunk count."""
    import faiss

    docs_dir = docs_dir or config.DOCS_DIR
    index_dir = index_dir or config.INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)

    reports = load_reports(docs_dir)
    chunks = build_chunks(reports)
    if not chunks:
        raise ValueError(f"No ingestible reports found under {docs_dir}")

    embeddings = get_embeddings()
    vectors = np.array(
        embeddings.embed_documents([c.text for c in chunks]), dtype="float32"
    )
    vectors = _normalize(vectors)

    faiss_index = faiss.IndexFlatIP(vectors.shape[1])
    faiss_index.add(vectors)
    faiss.write_index(faiss_index, str(index_dir / _INDEX_FILENAME))

    with open(index_dir / _METADATA_FILENAME, "w", encoding="utf-8") as fh:
        json.dump([asdict(c) for c in chunks], fh, indent=2)

    return len(chunks)


class Retriever:
    def __init__(self, index_dir: Path | None = None):
        import faiss

        self.index_dir = index_dir or config.INDEX_DIR
        index_path = self.index_dir / _INDEX_FILENAME
        metadata_path = self.index_dir / _METADATA_FILENAME
        if not index_path.exists() or not metadata_path.exists():
            build_index(index_dir=self.index_dir)

        self._faiss_index = faiss.read_index(str(index_path))
        with open(metadata_path, encoding="utf-8") as fh:
            raw_chunks = json.load(fh)
        self._chunks = [Chunk(**c) for c in raw_chunks]
        self._embeddings = get_embeddings()

    def search(self, query: str, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        top_k = top_k or config.RETRIEVAL_TOP_K
        query_vector = np.array([self._embeddings.embed_query(query)], dtype="float32")
        query_vector = _normalize(query_vector)
        scores, indices = self._faiss_index.search(query_vector, top_k)
        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._chunks[idx], float(score)))
        return results


_retriever_singleton: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = Retriever()
    return _retriever_singleton
