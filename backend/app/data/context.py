"""Build LLM-friendly textual context from a DataFrame.

The original CSV-AI shoved entire CSV text chunks into FAISS and retrieved
semantically. That's wasteful for small files and inaccurate for large
ones: a structured schema-plus-sample tends to beat raw retrieval for
analytical questions. We assemble three things:

  * Schema: name → dtype → null %.
  * Sample: a few representative rows.
  * Optional stats: numeric describe + missing-value summary.

The output is markdown, which most modern chat models handle well.
"""

from __future__ import annotations

from io import StringIO

import pandas as pd

from app.data.profiler import DataProfile, profile_dataframe
from app.data.sampler import head_tail_sample, smart_sample


def _schema_block(profile: DataProfile) -> str:
    lines = ["| column | dtype | missing | unique |", "|---|---|---|---|"]
    for col in profile.columns:
        lines.append(
            f"| {col.name} | {col.dtype} | {col.pct_missing:.1f}% | {col.n_unique} |"
        )
    return "\n".join(lines)


def _sample_block(df: pd.DataFrame, max_rows: int) -> str:
    sample = smart_sample(df, n=max_rows) if len(df) > max_rows else df
    buffer = StringIO()
    sample.to_csv(buffer, index=False)
    return buffer.getvalue().strip()


def build_dataframe_context(
    df: pd.DataFrame,
    *,
    profile: DataProfile | None = None,
    max_sample_rows: int = 30,
    include_correlation: bool = False,
) -> str:
    """Return a markdown string describing ``df`` for prompt injection."""

    profile = profile or profile_dataframe(df)

    parts = [
        "## Dataset overview",
        f"- Rows: {profile.n_rows}",
        f"- Columns: {profile.n_cols}",
        f"- Duplicate rows: {profile.duplicate_rows}",
        f"- Memory: {profile.memory_mb:.2f} MB",
        "",
        "## Schema",
        _schema_block(profile),
        "",
        "## Sample rows (CSV)",
        "```csv",
        _sample_block(df, max_sample_rows),
        "```",
    ]

    head_tail = head_tail_sample(df, head=3, tail=3)
    if len(head_tail) != len(df):
        parts += [
            "",
            "## Head and tail (CSV)",
            "```csv",
            head_tail.to_csv(index=False).strip(),
            "```",
        ]

    if include_correlation and profile.correlation is not None:
        parts += [
            "",
            "## Numeric correlation",
            "```",
            profile.correlation.round(2).to_string(),
            "```",
        ]

    return "\n".join(parts)
