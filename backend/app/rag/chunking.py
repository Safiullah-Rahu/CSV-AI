"""Chunking strategies.

Every strategy works on *segments* — runs of blocks that share a page,
section and sheet — so each chunk keeps an exact citation locator (page,
section trail, sheet + row range, slide, line range). Strategies:

``structure`` (default) · ``markdown`` · ``fixed`` · ``recursive`` · ``token``
· ``sentence`` · ``semantic`` · ``table`` (rows/tables kept intact) ·
``parent_child`` (small retrievable children inside larger context parents).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.ingestion.types import Block
from app.utils.tokens import estimate_tokens

STRATEGIES = ("structure", "markdown", "fixed", "recursive", "token", "sentence", "semantic", "table", "parent_child")
DEFAULT_CHUNKING: dict[str, Any] = {
    "strategy": "structure",
    "chunk_size": 1200,       # characters (tokens for the token strategy)
    "chunk_overlap": 150,
    "separators": ["\n\n", "\n", ". ", " ", ""],
    "contextual_headers": True,
    "parent_size": 3000,
    "child_size": 500,
    "semantic_percentile": 90,
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")
_ABBREV = re.compile(r"\b(?:e\.g|i\.e|etc|vs|Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|No|Fig|Sec|Art|approx|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.$")


@dataclass
class ChunkDraft:
    content: str
    meta: dict[str, Any] = field(default_factory=dict)
    parent: int | None = None
    retrievable: bool = True

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.content)


def merge_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = {**DEFAULT_CHUNKING, **(config or {})}
    if merged["strategy"] not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy '{merged['strategy']}'. Use one of {STRATEGIES}.")
    merged["chunk_size"] = max(100, int(merged["chunk_size"]))
    merged["chunk_overlap"] = max(0, min(int(merged["chunk_overlap"]), merged["chunk_size"] // 2))
    return merged


# --------------------------------------------------------------------------- helpers


def split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text)
    out: list[str] = []
    for part in parts:
        if out and _ABBREV.search(out[-1]):
            out[-1] = out[-1] + " " + part
        else:
            out.append(part)
    return [p.strip() for p in out if p.strip()]


def recursive_split(text: str, size: int, overlap: int, separators: Sequence[str]) -> list[str]:
    if len(text) <= size:
        return [text] if text.strip() else []
    sep = next((s for s in separators if s and s in text), "")
    if not sep:
        step = max(1, size - overlap)
        return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]
    pieces = text.split(sep)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = (current + sep + piece) if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(piece) > size:
            rest = [s for s in separators if s != sep]
            chunks.extend(recursive_split(piece, size, overlap, rest))
            current = ""
        else:
            tail = current[-overlap:] if overlap and current else ""
            current = (tail + sep + piece) if tail else piece
    if current.strip():
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


def token_split(text: str, size: int, overlap: int) -> list[str]:
    words = re.findall(r"\S+\s*", text)
    out, i = [], 0
    while i < len(words):
        window, count, j = [], 0, i
        while j < len(words) and count < size:
            window.append(words[j])
            count += max(1, estimate_tokens(words[j]))
            j += 1
        out.append("".join(window).strip())
        if j >= len(words):
            break
        back, k = 0, j
        while k > i + 1 and back < overlap:
            k -= 1
            back += max(1, estimate_tokens(words[k]))
        i = k if overlap else j
    return [o for o in out if o]


def sentence_pack(text: str, size: int, overlap_sentences: int = 1) -> list[str]:
    sentences = split_sentences(text)
    out: list[str] = []
    current: list[str] = []
    for s in sentences:
        if current and len(" ".join(current + [s])) > size:
            out.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
        if len(s) > size:
            out.extend(recursive_split(s, size, 0, [", ", " ", ""]))
            current = []
            continue
        current.append(s)
    if current:
        out.append(" ".join(current))
    return out


# -------------------------------------------------------------------------- segments


def _block_key(b: Block) -> tuple[Any, ...]:
    return (b.page, tuple(b.section[:3]), b.meta.get("sheet"), b.meta.get("slide"), b.meta.get("chapter"))


@dataclass
class Segment:
    blocks: list[Block]

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    def meta(self) -> dict[str, Any]:
        first = self.blocks[0]
        pages = [b.page for b in self.blocks if b.page is not None]
        meta: dict[str, Any] = {"section": first.section}
        if pages:
            meta["page"] = pages[0]
            if pages[-1] != pages[0]:
                meta["page_end"] = pages[-1]
        for key in ("sheet", "slide", "url", "chapter", "cell", "language"):
            if key in first.meta and first.meta[key] is not None:
                meta[key] = first.meta[key]
        rows = [b.meta for b in self.blocks if "row_start" in b.meta]
        if rows:
            meta["row_start"] = rows[0]["row_start"]
            meta["row_end"] = rows[-1]["row_end"]
        lines = [b.meta for b in self.blocks if "line_start" in b.meta]
        if lines:
            meta["line_start"] = lines[0]["line_start"]
            meta["line_end"] = lines[-1]["line_end"]
        tables = [b.meta["table_index"] for b in self.blocks if "table_index" in b.meta]
        if tables:
            meta["table_index"] = tables[0]
        kinds = {b.kind for b in self.blocks}
        meta["kind"] = next(iter(kinds)) if len(kinds) == 1 else "mixed"
        if any(b.meta.get("ocr") for b in self.blocks):
            meta["ocr"] = True
        return meta


def segments(blocks: Sequence[Block]) -> list[Segment]:
    out: list[Segment] = []
    for b in blocks:
        if out and _block_key(out[-1].blocks[-1]) == _block_key(b):
            out[-1].blocks.append(b)
        else:
            out.append(Segment([b]))
    return out


def _header(meta: dict[str, Any], title: str | None) -> str:
    parts = [p for p in ([title] if title else []) + list(meta.get("section") or []) if p]
    deduped = list(dict.fromkeys(parts))
    if meta.get("sheet") and meta["sheet"] not in deduped:
        deduped.append(f"Sheet {meta['sheet']}")
    return " › ".join(deduped)


def _with_header(text: str, meta: dict[str, Any], title: str | None, enabled: bool) -> str:
    header = _header(meta, title) if enabled else ""
    if header and not text.startswith(header):
        return f"[{header}]\n{text}"
    return text


# --------------------------------------------------------------------------- strategies


def _pack_blocks(seg: Segment, size: int, overlap: int, seps: Sequence[str]) -> list[tuple[str, list[Block]]]:
    """Greedy pack whole blocks up to ``size``; oversized blocks are split (tables by rows)."""
    out: list[tuple[str, list[Block]]] = []
    buf: list[Block] = []

    def flush() -> None:
        if buf:
            out.append(("\n\n".join(x.text for x in buf), list(buf)))
            buf.clear()

    for b in seg.blocks:
        if len(b.text) > size:
            flush()
            if b.kind in ("table", "rows"):
                lines = b.text.split("\n")
                header = lines[:2] if b.kind == "table" and len(lines) > 2 and set(lines[1].replace("|", "").strip()) <= {"-", " "} else []
                body = lines[len(header):]
                part: list[str] = []
                for line in body:
                    if part and len("\n".join(header + part + [line])) > size:
                        out.append(("\n".join(header + part), [b]))
                        part = []
                    part.append(line)
                if part:
                    out.append(("\n".join(header + part), [b]))
            else:
                for piece in recursive_split(b.text, size, overlap, seps):
                    out.append((piece, [b]))
            continue
        if buf and len("\n\n".join(x.text for x in buf + [b])) > size:
            flush()
        buf.append(b)
    flush()
    return out


def _segment_meta_for(blocks: list[Block]) -> dict[str, Any]:
    return Segment(blocks).meta()


def chunk_blocks(
    blocks: Sequence[Block],
    config: dict[str, Any] | None = None,
    *,
    title: str | None = None,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[ChunkDraft]:
    cfg = merge_config(config)
    strategy = cfg["strategy"]
    size, overlap, seps = cfg["chunk_size"], cfg["chunk_overlap"], cfg["separators"]
    headers = bool(cfg["contextual_headers"])
    segs = segments([b for b in blocks if b.kind != "heading" or strategy in ("fixed", "recursive", "token", "sentence")])
    drafts: list[ChunkDraft] = []

    if strategy == "parent_child":
        for seg in segs:
            for parent_text, parent_blocks in _pack_blocks(seg, int(cfg["parent_size"]), 0, seps):
                meta = _segment_meta_for(parent_blocks)
                drafts.append(ChunkDraft(_with_header(parent_text, meta, title, headers), meta, retrievable=False))
                parent_index = len(drafts) - 1
                for child in recursive_split(parent_text, int(cfg["child_size"]), overlap // 2, seps):
                    drafts.append(ChunkDraft(_with_header(child, meta, title, headers), dict(meta), parent=parent_index))
        return drafts

    for seg in segs:
        meta = seg.meta()
        if strategy in ("structure", "markdown", "table"):
            for text, used in _pack_blocks(seg, size, overlap, seps):
                m = _segment_meta_for(used)
                drafts.append(ChunkDraft(_with_header(text, m, title, headers), m))
            continue
        text = seg.text
        if strategy == "fixed":
            step = max(1, size - overlap)
            pieces = [text[i : i + size] for i in range(0, len(text), step)]
        elif strategy == "recursive":
            pieces = recursive_split(text, size, overlap, seps)
        elif strategy == "token":
            pieces = token_split(text, size, overlap)
        elif strategy == "sentence":
            pieces = sentence_pack(text, size)
        elif strategy == "semantic":
            pieces = semantic_split(text, size, embed, int(cfg["semantic_percentile"]))
        else:  # pragma: no cover - guarded by merge_config
            pieces = [text]
        for piece in pieces:
            if piece.strip():
                drafts.append(ChunkDraft(_with_header(piece.strip(), meta, title, headers), dict(meta)))
    return drafts


def semantic_split(
    text: str,
    size: int,
    embed: Callable[[list[str]], list[list[float]]] | None,
    percentile: int = 90,
) -> list[str]:
    """Break where adjacent-sentence embedding distance spikes (percentile threshold)."""
    sentences = split_sentences(text)
    if embed is None or len(sentences) < 4:
        return sentence_pack(text, size)
    vectors = np.asarray(embed(sentences), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norms, 1e-9, None)
    distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)
    threshold = float(np.percentile(distances, percentile))
    groups: list[list[str]] = [[sentences[0]]]
    for sentence, dist in zip(sentences[1:], distances):
        if dist >= threshold or len(" ".join(groups[-1] + [sentence])) > size:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)
    out: list[str] = []
    for g in groups:
        joined = " ".join(g)
        out.extend(recursive_split(joined, size, 0, [". ", " ", ""]) if len(joined) > size else [joined])
    return out
