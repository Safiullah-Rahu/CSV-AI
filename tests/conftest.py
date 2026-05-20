"""Shared pytest fixtures."""

from __future__ import annotations

import io

import pandas as pd
import pytest


@pytest.fixture()
def sample_csv_bytes() -> bytes:
    """A tiny but realistic CSV with mixed dtypes and a missing value."""
    return (
        b"id,name,score,category\n"
        b"1,Alice,9.5,A\n"
        b"2,Bob,7.0,B\n"
        b"3,Charlie,,A\n"
        b"4,Dana,8.1,B\n"
    )


@pytest.fixture()
def sample_df(sample_csv_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(sample_csv_bytes))
