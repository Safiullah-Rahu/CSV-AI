"""Content normalisation, language detection, auto-tagging and injection screening."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from app.ingestion.types import Block, ParsedDocument

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_SPACES = re.compile(r"[ \t\u00a0]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")

STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "his", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or", "our", "she", "should", "so", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "to", "too", "us", "was", "we", "were", "what", "when", "where", "which", "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "about", "above", "after", "again", "against", "all", "am", "any", "because", "before", "being", "below", "between", "both", "down", "during", "each", "few", "further", "here", "more", "most", "other", "out", "over", "own", "same", "some", "such", "through", "under", "until", "up", "very", "also", "may", "must", "shall", "per", "via", "etc"]
)

# Phrases typical of prompt-injection payloads hidden in documents or web pages.
_INJECTION = [
    re.compile(p, re.I)
    for p in (
        r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|prompts|rules|messages)",
        r"disregard (?:all |any |the )?(?:previous|prior|above|system) (?:instructions|prompts|rules)",
        r"forget (?:all |everything |your )(?:previous |prior )?(?:instructions|rules|training)",
        r"you are now (?:a|an|in) ",
        r"(?:reveal|print|show|output|repeat) (?:your|the) (?:system prompt|instructions|hidden prompt)",
        r"(?:system|developer) (?:prompt|message)\s*[:=]",
        r"<\|?(?:im_start|im_end|system|endoftext)\|?>",
        r"\[\s*(?:system|INST)\s*\]",
        r"new instructions\s*:",
        r"(?:send|exfiltrate|upload|post) (?:the |all )?(?:data|conversation|secrets|credentials|api keys?) to",
        r"!\[[^\]]*\]\(https?://[^)]*\?[^)]*=",  # markdown image beacon with query string
    )
]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_SPACES.sub(" ", line).rstrip() for line in text.split("\n"))
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def normalize_document(doc: ParsedDocument) -> ParsedDocument:
    blocks: list[Block] = []
    for b in doc.blocks:
        text = normalize_text(b.text) if b.kind not in ("code",) else _CONTROL.sub("", b.text).replace("\r\n", "\n").strip("\n")
        if text.strip():
            b.text = text
            b.section = [normalize_text(s)[:200] for s in b.section if s]
            blocks.append(b)
    doc.blocks = blocks
    return doc


def detect_language(text: str) -> str | None:
    sample = text[:3000]
    if len(re.findall(r"\w", sample)) < 40:
        return None
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(sample)
    except Exception:  # noqa: BLE001
        return None


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def auto_tags(text: str, limit: int = 6) -> list[str]:
    words = [w for w in tokenize(text[:200_000]) if len(w) > 3 and w not in STOPWORDS and not w.isdigit()]
    if not words:
        return []
    counts = Counter(words)
    # Prefer terms that are frequent but not ubiquitous noise.
    return [w for w, c in counts.most_common(limit * 3) if c >= 3][:limit]


def injection_findings(text: str) -> list[str]:
    hits = []
    for pattern in _INJECTION:
        m = pattern.search(text)
        if m:
            hits.append(m.group(0)[:120])
    return hits


def neutralize(text: str) -> str:
    """Bracket instruction-like spans so the model sees them as quoted data."""
    for pattern in _INJECTION:
        text = pattern.sub(lambda m: f"[quoted text: {m.group(0)}]", text)
    return text
