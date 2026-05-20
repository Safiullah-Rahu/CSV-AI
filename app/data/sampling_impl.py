"""Smart sampling strategies for large CSVs (implementation module).

Stratified sampling on the most-balanced low-cardinality categorical
column tends to outperform naive random sampling for analytical QA tasks
because it ensures the model sees every important category.
"""

from __future__ import annotations

import pandas as pd


def smart_sample(
    df: pd.DataFrame,
    n: int = 200,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return at most ``n`` rows from ``df`` using stratification if possible."""
    if len(df) <= n:
        return df.copy()

    candidates = [
        c
        for c in df.select_dtypes(include=["object", "category", "bool"]).columns
        if 1 < df[c].nunique(dropna=True) <= 20
    ]
    candidates.sort(key=lambda c: df[c].nunique(dropna=True))

    if candidates:
        try:
            return _stratify(df, candidates[0], n, random_state)
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
    """Return head + tail rows for compact prompts."""
    if len(df) <= head + tail:
        return df.copy()
    return pd.concat([df.head(head), df.tail(tail)])
