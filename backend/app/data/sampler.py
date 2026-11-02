"""Smart sampling for large CSVs.

Stratified sampling on a low-cardinality categorical column tends to
outperform naive random sampling for analytical QA. We fall back to
uniform random sampling when no suitable key exists.
"""

from __future__ import annotations

import pandas as pd


def smart_sample(df: pd.DataFrame, n: int = 200, *, random_state: int = 42) -> pd.DataFrame:
    """Return up to ``n`` rows, stratified when possible."""
    if len(df) <= n:
        return df.copy()

    cands = [
        c
        for c in df.select_dtypes(include=["object", "category", "bool"]).columns
        if 1 < df[c].nunique(dropna=True) <= 20
    ]
    cands.sort(key=lambda c: df[c].nunique(dropna=True))

    if cands:
        try:
            return _stratify(df, cands[0], n, random_state)
        except Exception:
            pass

    return df.sample(n=n, random_state=random_state)


def _stratify(df: pd.DataFrame, key: str, n: int, rs: int) -> pd.DataFrame:
    """Group-by stratified sample, robust across pandas versions."""

    def draw(g: pd.DataFrame) -> pd.DataFrame:
        size = max(1, int(round(n * len(g) / len(df))))
        return g.sample(n=min(size, len(g)), random_state=rs)

    try:
        out = df.groupby(key, group_keys=False, dropna=False).apply(
            draw, include_groups=False
        )
        return df.loc[out.index].head(n)
    except TypeError:
        return df.groupby(key, group_keys=False, dropna=False).apply(draw).head(n)


def head_tail_sample(df: pd.DataFrame, head: int = 5, tail: int = 5) -> pd.DataFrame:
    """First + last rows for compact prompts."""
    if len(df) <= head + tail:
        return df.copy()
    return pd.concat([df.head(head), df.tail(tail)])


# Re-exports kept tidy for ``from app.data.sampler import *`` callers.
__all__ = ["smart_sample", "head_tail_sample"]
