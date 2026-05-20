"""Statistical profiling of a DataFrame.

Returns a structured :class:`DataProfile` that the UI can render directly
and the LLM can consume as JSON-shaped context. Avoids pandas-profiling-
style heavyweight reports; we want something fast enough to recompute on
every upload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_missing: int
    pct_missing: float
    n_unique: int
    sample_values: List[Any] = field(default_factory=list)
    # Numeric-only fields
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None


@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    memory_mb: float
    columns: List[ColumnProfile]
    duplicate_rows: int
    correlation: Optional[pd.DataFrame] = None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a small JSON-friendly summary safe to inject into prompts."""
        return {
            "rows": self.n_rows,
            "cols": self.n_cols,
            "duplicate_rows": self.duplicate_rows,
            "memory_mb": round(self.memory_mb, 3),
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "missing_pct": round(c.pct_missing, 2),
                    "unique": c.n_unique,
                }
                for c in self.columns
            ],
        }


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def profile_dataframe(df: pd.DataFrame, *, sample_size: int = 3) -> DataProfile:
    """Compute summary statistics for ``df``.

    ``sample_size`` controls how many example values we capture per column —
    useful for showing the LLM what the data actually looks like.
    """

    columns: List[ColumnProfile] = []
    for name in df.columns:
        series = df[name]
        n_missing = int(series.isna().sum())
        pct_missing = (n_missing / len(df) * 100) if len(df) else 0.0
        try:
            n_unique = int(series.nunique(dropna=True))
        except TypeError:
            # Unhashable types (e.g. lists) — fall back to a length proxy.
            n_unique = -1

        col = ColumnProfile(
            name=str(name),
            dtype=str(series.dtype),
            n_missing=n_missing,
            pct_missing=pct_missing,
            n_unique=n_unique,
        )

        sample = series.dropna().head(sample_size).tolist()
        col.sample_values = [str(v) for v in sample]

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            col.mean = _safe_float(desc.get("mean"))
            col.std = _safe_float(desc.get("std"))
            col.min = _safe_float(desc.get("min"))
            col.max = _safe_float(desc.get("max"))
            col.median = _safe_float(series.median(skipna=True))

        columns.append(col)

    # Correlation: numeric columns only, suppressed if too few.
    numeric_df = df.select_dtypes(include="number")
    correlation = numeric_df.corr() if numeric_df.shape[1] >= 2 else None

    return DataProfile(
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        memory_mb=float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
        columns=columns,
        duplicate_rows=int(df.duplicated().sum()),
        correlation=correlation,
    )
