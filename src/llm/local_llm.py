"""Local, non-paid LLM and embedding client factory.

Per constitution Principle IV, no paid LLM API may be used. The default path
serves both chat and embeddings from a local Ollama instance. If Ollama is
unavailable, set USE_DUMMY_LLM=1 to fall back to a deterministic mock model
that still lets the rest of the graph run end-to-end.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from typing import Protocol


class ChatModel(Protocol):
    def invoke(self, prompt: str) -> str: ...


class EmbeddingModel(Protocol):
    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class _DummyChatModel:
    """Deterministic stand-in LLM used when local inference is infeasible."""

    def invoke(self, prompt: str) -> str:
        return (
            "[dummy-llm] I do not have a real model configured, but here is a "
            "placeholder response based on the given context.\n\n" + prompt[-500:]
        )


class _DummyEmbeddingModel:
    """Deterministic hash-based embedding, only for offline/dummy mode."""

    _dims = 64

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self._dims]]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


def _use_dummy() -> bool:
    return os.environ.get("USE_DUMMY_LLM", "").lower() in {"1", "true", "yes"}


@lru_cache(maxsize=1)
def get_chat_llm() -> ChatModel:
    if _use_dummy():
        return _DummyChatModel()

    from langchain_ollama import ChatOllama

    from src import config

    return ChatOllama(
        base_url=config.OLLAMA_BASE_URL,
        model=config.LLM_MODEL,
        temperature=0.1,
        client_kwargs={"timeout": config.LLM_TIMEOUT_SECONDS},
    )


@lru_cache(maxsize=1)
def get_embeddings() -> EmbeddingModel:
    if _use_dummy():
        return _DummyEmbeddingModel()

    from langchain_ollama import OllamaEmbeddings

    from src import config

    return OllamaEmbeddings(base_url=config.OLLAMA_BASE_URL, model=config.EMBEDDING_MODEL)


def invoke_llm(prompt: str) -> str:
    """Invoke the configured chat model and normalize the response to plain text."""
    llm = get_chat_llm()
    result = llm.invoke(prompt)
    # langchain chat models return a BaseMessage-like object with .content;
    # the dummy model returns a plain string.
    return getattr(result, "content", result)
