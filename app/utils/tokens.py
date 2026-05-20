"""Token-counting helpers.

Uses ``tiktoken`` when available; falls back to a coarse character-based
estimate so the rest of the app does not depend on the tokenizer being
present (useful for local providers).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=8)
def _get_encoder(model: str):  # pragma: no cover - thin wrapper
    try:
        import tiktoken

        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def estimate_tokens(text: str, model: Optional[str] = None) -> int:
    """Estimate the number of tokens for ``text``.

    Uses tiktoken for OpenAI-family models; otherwise approximates one token
    per four characters (a well-known rough heuristic).
    """
    if not text:
        return 0

    encoder = _get_encoder(model or "gpt-4o-mini")
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass

    # Fallback heuristic: ~4 chars / token for English text.
    return max(1, len(text) // 4)
