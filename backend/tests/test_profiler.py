"""Tests for the dataframe profiler."""

from __future__ import annotations

import pandas as pd

from app.data.context import build_dataframe_context
from app.data.profiler import profile_dataframe
from app.data.sampler import smart_sample


def test_profile_basic(sample_df: pd.DataFrame) -> None:
    profile = profile_dataframe(sample_df)
    assert profile.n_rows == 4
    assert profile.n_cols == 4

    score = next(c for c in profile.columns if c.name == "score")
    assert score.n_missing == 1
    assert score.pct_missing == 25.0
    assert score.mean is not None


def test_profile_correlation_requires_two_numeric() -> None:
    df_single = pd.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "c"]})
    profile = profile_dataframe(df_single)
    assert profile.correlation is None


def test_smart_sample_preserves_small_frames(sample_df: pd.DataFrame) -> None:
    out = smart_sample(sample_df, n=100)
    assert len(out) == len(sample_df)


def test_smart_sample_reduces_large_frames() -> None:
    df = pd.DataFrame({"a": list(range(1000)), "g": ["x", "y"] * 500})
    out = smart_sample(df, n=50)
    assert len(out) <= 50


def test_context_contains_schema_and_sample(sample_df: pd.DataFrame) -> None:
    text = build_dataframe_context(sample_df)
    assert "## Schema" in text
    assert "## Sample rows" in text
    assert "score" in text
