"""Parser output model shared by every format."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Block:
    """A structural unit of a document with its citation locator.

    ``kind``: heading | paragraph | table | code | rows | list | slide | cell | text
    ``section``: heading trail, e.g. ["Leave policy", "Annual leave"].
    ``meta``: sheet, row_start/row_end, table_index, slide, line_start/line_end, url, cell.
    """

    text: str
    kind: str = "paragraph"
    page: int | None = None
    section: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TabularSource:
    """A table the dataset engine can query with SQL (CSV, sheet, Parquet, JSON array)."""

    name: str
    path: str  # local path of a Parquet file written by the parser
    rows: int
    columns: list[str]


@dataclass
class ParsedDocument:
    blocks: list[Block]
    parser: str
    title: str | None = None
    page_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    tables: list[TabularSource] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text)


class UnsupportedFormat(Exception):
    """The format is unknown, disabled, or needs an optional dependency."""


class InvalidDocument(Exception):
    """The file is corrupt, mislabelled or rejected by a safety check."""
