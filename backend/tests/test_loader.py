"""Tests for the CSV loader."""

from __future__ import annotations

import io

import pytest

from app.data.loader import load_csv
from app.utils.errors import DataError


def test_loads_utf8_csv(sample_csv_bytes: bytes) -> None:
    dataset = load_csv(sample_csv_bytes, source_name="sample.csv")
    assert dataset.n_rows == 4
    assert dataset.n_cols == 4
    assert dataset.encoding in {"utf-8", "utf-8-sig"}
    assert dataset.delimiter == ","


def test_loads_semicolon_delimited() -> None:
    raw = b"a;b;c\n1;2;3\n4;5;6\n"
    dataset = load_csv(raw)
    assert dataset.n_rows == 2
    assert dataset.delimiter == ";"


def test_loads_cp1252_with_smart_quotes() -> None:
    raw = "name,quote\nAlice,“hello”\n".encode("cp1252")
    dataset = load_csv(raw)
    assert dataset.n_rows == 1
    assert "hello" in dataset.df.iloc[0]["quote"]


def test_rejects_empty_input() -> None:
    with pytest.raises(DataError):
        load_csv(b"")


def test_accepts_filelike() -> None:
    buf = io.BytesIO(b"a,b\n1,2\n3,4\n")
    dataset = load_csv(buf, source_name="buf.csv")
    assert dataset.n_rows == 2
