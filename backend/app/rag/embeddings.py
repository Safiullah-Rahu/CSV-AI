"""Embedding providers behind one interface.

``hashing`` is a real, dependency-free baseline (signed feature hashing of
word unigrams + bigrams + character trigrams, sublinear TF, L2 norm). It is
lexical rather than semantic — fine for self-hosted installs without a
model, and deterministic for tests. Production knowledge bases should use a
neural model (OpenAI, Cohere, Voyage, Ollama, local ONNX via fastembed,
Hugging Face TEI, or any custom endpoint).
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from functools import lru_cache
from typing import Any, Protocol

import httpx
import numpy as np

from app.utils.errors import ConfigurationError, NotAvailable, ProviderError

_WORD = re.compile(r"\w+")


class Embedder(Protocol):
    provider: str
    model: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbedder:
    provider = "hashing"
    # Cosine similarity that unrelated texts typically reach with this model (evidence scaling).
    similarity_floor = 0.0

    def __init__(self, dim: int = 768) -> None:
        if not 64 <= dim <= 4096:
            raise ConfigurationError("Hashing embedder dimension must be 64–4096.")
        self.dim = dim
        self.model = f"hashing-{dim}"

    @staticmethod
    def _h(feature: str) -> int:
        return int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "little")

    def _features(self, text: str) -> Counter[str]:
        words = [w for w in _WORD.findall(text.lower()) if len(w) > 1]
        feats: Counter[str] = Counter(f"w:{w}" for w in words)
        feats.update(f"b:{a}_{b}" for a, b in zip(words, words[1:]))
        for w in words:
            if len(w) > 4:
                padded = f"#{w}#"
                feats.update(f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2))
        return feats

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat, count in self._features(text).items():
            h = self._h(feat)
            weight = (1.0 + math.log(count)) * (0.5 if feat[0] == "c" else 1.0)
            vec[h % self.dim] += weight if (h >> 63) & 1 == 0 else -weight
        norm = float(np.linalg.norm(vec))
        return (vec / norm).tolist() if norm > 0 else vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class _HTTPEmbedder:
    provider = "http"
    batch_size = 64
    similarity_floor = 0.2

    def __init__(self, model: str, dim: int | None, *, api_key: str | None, base_url: str | None, timeout: float = 60.0, headers: dict[str, str] | None = None) -> None:
        self.model = model
        self.dim = dim or 0
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout
        self.extra_headers = headers or {}

    def _post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        last: Exception | None = None
        for attempt in range(4):
            try:
                r = httpx.post(url, json=payload, headers={**headers, **self.extra_headers}, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise ProviderError(f"{self.provider} embeddings HTTP {r.status_code}")
                if r.status_code >= 400:
                    raise ProviderError(f"{self.provider} embeddings rejected the request (HTTP {r.status_code}): {r.text[:300]}")
                return r.json()
            except (httpx.HTTPError, ProviderError) as exc:
                last = exc
                if isinstance(exc, ProviderError) and "rejected" in str(exc):
                    raise
                time.sleep(min(8, 0.5 * 2**attempt))
        raise ProviderError(f"{self.provider} embeddings failed: {last}")

    def _check(self, vectors: list[list[float]]) -> list[list[float]]:
        if vectors and self.dim and len(vectors[0]) != self.dim:
            raise ConfigurationError(f"Embedding dimension mismatch: model returned {len(vectors[0])}, knowledge base expects {self.dim}.")
        if vectors and not self.dim:
            self.dim = len(vectors[0])
        return vectors

    def _batch(self, texts: list[str], kind: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._call(texts[i : i + self.batch_size], kind))
        return self._check(out)

    def _call(self, texts: list[str], kind: str) -> list[list[float]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._batch(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        return self._batch([text], "query")[0]


class OpenAIEmbedder(_HTTPEmbedder):
    """OpenAI, Azure OpenAI (``api_version`` + deployment URL) and OpenAI-compatible servers (vLLM, LocalAI, LM Studio)."""

    provider = "openai"

    def __init__(self, *args: Any, api_version: str | None = None, azure: bool = False, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.api_version = api_version
        self.azure = azure
        if not self.base_url:
            self.base_url = "https://api.openai.com/v1"

    def _call(self, texts: list[str], kind: str) -> list[list[float]]:
        if self.azure:
            url = f"{self.base_url}/openai/deployments/{self.model}/embeddings?api-version={self.api_version or '2024-06-01'}"
            headers = {"api-key": self.api_key or ""}
        else:
            url = f"{self.base_url}/embeddings"
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload: dict[str, Any] = {"input": texts, "model": self.model}
        data = self._post(url, payload, headers)
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]


class CohereEmbedder(_HTTPEmbedder):
    provider = "cohere"
    batch_size = 96

    def _call(self, texts: list[str], kind: str) -> list[list[float]]:
        url = f"{self.base_url or 'https://api.cohere.com'}/v2/embed"
        payload = {"model": self.model, "texts": texts, "input_type": "search_query" if kind == "query" else "search_document", "embedding_types": ["float"]}
        data = self._post(url, payload, {"Authorization": f"Bearer {self.api_key}"})
        return data["embeddings"]["float"]


class VoyageEmbedder(_HTTPEmbedder):
    provider = "voyage"
    batch_size = 128

    def _call(self, texts: list[str], kind: str) -> list[list[float]]:
        url = f"{self.base_url or 'https://api.voyageai.com/v1'}/embeddings"
        data = self._post(url, {"model": self.model, "input": texts, "input_type": kind}, {"Authorization": f"Bearer {self.api_key}"})
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]


class OllamaEmbedder(_HTTPEmbedder):
    provider = "ollama"
    batch_size = 32

    def _call(self, texts: list[str], kind: str) -> list[list[float]]:
        data = self._post(f"{self.base_url or 'http://localhost:11434'}/api/embed", {"model": self.model, "input": texts}, {})
        return data["embeddings"]


class HuggingFaceEmbedder(_HTTPEmbedder):
    """Hugging Face Inference endpoints / Text Embeddings Inference (``POST {inputs}``)."""

    provider = "huggingface"

    def _call(self, texts: list[str], kind: str) -> list[list[float]]:
        url = self.base_url or f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = self._post(url, {"inputs": texts, "options": {"wait_for_model": True}}, headers)
        vectors = []
        for item in data:
            arr = np.asarray(item, dtype=np.float32)
            vectors.append((arr.mean(axis=0) if arr.ndim == 2 else arr).tolist())
        return vectors


class CustomEmbedder(_HTTPEmbedder):
    """Any endpoint accepting ``{"texts": [...], "model", "input_type"}`` and returning ``{"embeddings": [[...]]}``."""

    provider = "custom"

    def _call(self, texts: list[str], kind: str) -> list[list[float]]:
        if not self.base_url:
            raise ConfigurationError("Custom embedding endpoint URL is required.")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return self._post(self.base_url, {"texts": texts, "model": self.model, "input_type": kind}, headers)["embeddings"]


class FastEmbedEmbedder:
    """Local ONNX sentence-embedding models (bge, MiniLM, e5) via ``fastembed`` — CPU, offline after download."""

    provider = "local"
    similarity_floor = 0.45  # bge/MiniLM cosine for unrelated sentences sits around 0.4-0.55

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", dim: int | None = None) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise NotAvailable("Local embeddings need the 'local-embeddings' extra (pip install fastembed).") from exc
        self.model = model
        self._model = _load_fastembed(model, TextEmbedding)
        probe = next(iter(self._model.embed(["dimension probe"])))
        self.dim = len(probe)
        if dim and dim != self.dim:
            raise ConfigurationError(f"Model {model} produces {self.dim}-d vectors, not {dim}.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        fn = getattr(self._model, "query_embed", None)
        vec = next(iter(fn([text]))) if fn else next(iter(self._model.embed([text])))
        return vec.tolist()


@lru_cache(maxsize=4)
def _load_fastembed(model: str, cls: Any) -> Any:
    return cls(model_name=model)


EMBEDDING_PROVIDERS = {
    "hashing": "Lexical hashing (offline baseline, no model)",
    "openai": "OpenAI",
    "azure_openai": "Azure OpenAI",
    "openai_compatible": "OpenAI-compatible endpoint (vLLM, LocalAI, LM Studio)",
    "cohere": "Cohere",
    "voyage": "Voyage AI",
    "huggingface": "Hugging Face Inference / TEI",
    "ollama": "Ollama",
    "local": "Local ONNX model (fastembed: bge, MiniLM, e5)",
    "custom": "Custom HTTP endpoint",
}


def build_embedder(config: dict[str, Any], *, api_key: str | None = None, base_url: str | None = None, settings: dict[str, Any] | None = None) -> Embedder:
    provider = config.get("provider", "hashing")
    model = config.get("model") or ""
    dim = config.get("dim")
    settings = settings or {}
    if provider == "hashing":
        return HashingEmbedder(int(dim or 768))
    if provider == "local":
        return FastEmbedEmbedder(model or "BAAI/bge-small-en-v1.5", dim)
    kwargs = {"api_key": api_key, "base_url": base_url or config.get("base_url")}
    if provider in ("openai", "openai_compatible"):
        return OpenAIEmbedder(model or "text-embedding-3-small", dim, **kwargs)
    if provider == "azure_openai":
        return OpenAIEmbedder(model, dim, azure=True, api_version=settings.get("api_version"), **kwargs)
    if provider == "cohere":
        return CohereEmbedder(model or "embed-english-v3.0", dim, **kwargs)
    if provider == "voyage":
        return VoyageEmbedder(model or "voyage-3", dim, **kwargs)
    if provider == "ollama":
        return OllamaEmbedder(model or "nomic-embed-text", dim, **kwargs)
    if provider == "huggingface":
        return HuggingFaceEmbedder(model, dim, **kwargs)
    if provider == "custom":
        return CustomEmbedder(model or "custom", dim, **kwargs)
    raise ConfigurationError(f"Unknown embedding provider '{provider}'.")
