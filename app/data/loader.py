"""Robust CSV loader.

Handles encoding detection (utf-8 → cp1252 → latin-1 fallback chain),
delimiter sniffing, and returns a typed dataclass so downstream code does
not need to juggle ``Optional[DataFrame]`` everywhere.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import IO, Optional, Tuple, Union

import pandas as pd

from app.utils.errors import DataError
from app.utils.logging import get_logger

logger = get_logger("data.loader")

_ENCODINGS: Tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


@dataclass
class LoadedDataset:
    """A successfully-loaded CSV plus the metadata we sniffed."""

    df: pd.DataFrame
    encoding: str
    delimiter: str
    source_name: str
    n_rows: int
    n_cols: int


def _read_bytes(source: Union[str, bytes, IO[bytes]]) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, str):
        with open(source, "rb") as handle:
            return handle.read()
    if hasattr(source, "read"):
        data = source.read()
        # Streamlit's UploadedFile.read() returns bytes; rewind for reuse.
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass
        if isinstance(data, str):
            data = data.encode("utf-8")
        return data
    raise DataError(f"Unsupported CSV source: {type(source)!r}")


def _detect_encoding(raw: bytes) -> str:
    """Try each candidate encoding until one decodes the first 4 KiB cleanly."""
    sample = raw[:4096]
    for encoding in _ENCODINGS:
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"  # Always succeeds; lossy but safe.


def _detect_delimiter(text_sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(text_sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def load_csv(
    source: Union[str, bytes, IO[bytes]],
    *,
    source_name: Optional[str] = None,
    nrows: Optional[int] = None,
) -> LoadedDataset:
    """Load a CSV from a path, bytes, or file-like object.

    Raises :class:`DataError` on failure with a readable message. Never
    silently swallows decoding errors.
    """

    raw = _read_bytes(source)
    if not raw:
        raise DataError("Uploaded file is empty.")

    encoding = _detect_encoding(raw)
    try:
        text_sample = raw[:8192].decode(encoding, errors="replace")
    except Exception as exc:  # pragma: no cover - defensive
        raise DataError(f"Could not decode file as text: {exc}") from exc

    delimiter = _detect_delimiter(text_sample)

    try:
        df = pd.read_csv(
            io.BytesIO(raw),
            encoding=encoding,
            sep=delimiter,
            nrows=nrows,
            engine="python",  # tolerates ragged rows better than the C engine
            on_bad_lines="skip",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("CSV parse failed")
        raise DataError(f"Failed to parse CSV: {exc}") from exc

    name = source_name or (source if isinstance(source, str) else "uploaded.csv")

    return LoadedDataset(
        df=df,
        encoding=encoding,
        delimiter=delimiter,
        source_name=str(name),
        n_rows=len(df),
        n_cols=df.shape[1],
    )
