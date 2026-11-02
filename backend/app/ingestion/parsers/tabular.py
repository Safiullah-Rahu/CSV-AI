"""Structured data: CSV/TSV, Excel, OpenDocument spreadsheets, JSON/JSONL, Parquet.

Each table becomes (a) row blocks for retrieval — "Row 12 — name: Alice; ..." —
with sheet/row locators for citations, and (b) a Parquet copy the dataset
engine can query with SQL (CSV-AI's structured-analysis capability).
"""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from collections.abc import Iterable
from typing import Any

import pandas as pd

from app.data.loader import _detect_delimiter, _detect_encoding
from app.ingestion.types import Block, InvalidDocument, ParsedDocument, TabularSource

ROWS_PER_BLOCK = 15
MAX_ROWS_AS_TEXT = 20_000
MAX_ROWS_TABLE = 5_000_000


def _cell(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    return s[:500]


def rows_to_blocks(columns: list[str], rows: Iterable[tuple[int, list[Any]]], *, sheet: str | None, warnings: list[str]) -> list[Block]:
    blocks: list[Block] = []
    batch: list[str] = []
    start = None
    last = 0
    for index, values in rows:
        if index > MAX_ROWS_AS_TEXT:
            warnings.append(f"{sheet or 'table'}: only the first {MAX_ROWS_AS_TEXT} rows are indexed as text; all rows remain queryable with SQL.")
            break
        pairs = [f"{c}: {_cell(v)}" for c, v in zip(columns, values) if _cell(v) != ""]
        if not pairs:
            continue
        start = index if start is None else start
        last = index
        batch.append(f"Row {index} — " + "; ".join(pairs))
        if len(batch) >= ROWS_PER_BLOCK:
            blocks.append(Block(text="\n".join(batch), kind="rows", section=[sheet] if sheet else [], meta={"sheet": sheet, "row_start": start, "row_end": last, "columns": columns[:50]}))
            batch, start = [], None
    if batch:
        blocks.append(Block(text="\n".join(batch), kind="rows", section=[sheet] if sheet else [], meta={"sheet": sheet, "row_start": start, "row_end": last, "columns": columns[:50]}))
    return blocks


def _clean_columns(cols: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: dict[str, int] = {}
    for i, c in enumerate(cols):
        name = re.sub(r"\s+", " ", str(c if c is not None and str(c).strip() else f"column_{i + 1}")).strip()[:120]
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def _write_parquet(df: pd.DataFrame) -> str:
    fd, path = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
    df.to_parquet(path, index=False)
    return path


def table_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "table"
    return ("t_" + slug if slug[0].isdigit() else slug)[:60]


# ----------------------------------------------------------------------------- CSV


def parse_delimited(path: str, ext: str) -> ParsedDocument:
    with open(path, "rb") as fh:
        head = fh.read(65536)
    encoding = _detect_encoding(head)
    delimiter = "\t" if ext == "tsv" else _detect_delimiter(head[:8192].decode(encoding, errors="replace"))
    warnings: list[str] = []
    blocks: list[Block] = []
    frames: list[pd.DataFrame] = []
    total = 0
    try:
        reader = pd.read_csv(path, sep=delimiter, encoding=encoding, chunksize=50_000, engine="python", on_bad_lines="skip", dtype_backend="numpy_nullable")
        columns: list[str] | None = None
        for chunk in reader:
            if columns is None:
                columns = _clean_columns(chunk.columns)
            chunk.columns = columns
            if total < MAX_ROWS_AS_TEXT:
                rows = ((total + i + 1, list(r)) for i, r in enumerate(chunk.itertuples(index=False, name=None)))
                blocks.extend(rows_to_blocks(columns, rows, sheet=None, warnings=warnings))
            frames.append(chunk)
            total += len(chunk)
            if total >= MAX_ROWS_TABLE:
                warnings.append(f"Table truncated at {MAX_ROWS_TABLE} rows.")
                break
    except (pd.errors.ParserError, UnicodeDecodeError, csv.Error) as exc:
        raise InvalidDocument(f"Could not parse delimited file: {exc}") from exc
    if not frames:
        raise InvalidDocument("The file contains no rows.")
    df = pd.concat(frames, ignore_index=True)
    base = os.path.splitext(os.path.basename(path))[0]
    tables = [TabularSource(name=table_name(base), path=_write_parquet(df), rows=len(df), columns=list(df.columns))]
    meta = {"encoding": encoding, "delimiter": delimiter, "rows": len(df), "columns": len(df.columns)}
    return ParsedDocument(blocks=blocks, parser="csv", metadata=meta, warnings=_dedupe(warnings), tables=tables, page_count=None)


# ------------------------------------------------------------------------- spreadsheets


def _sheet_to_doc(name: str, rows: list[list[Any]], warnings: list[str]) -> tuple[list[Block], TabularSource | None]:
    rows = [r for r in rows if any(_cell(v) for v in r)]
    if not rows:
        return [], None
    columns = _clean_columns(rows[0])
    body = [list(r) + [None] * (len(columns) - len(r)) for r in rows[1:]]
    body = [r[: len(columns)] for r in body]
    blocks = rows_to_blocks(columns, ((i + 2, r) for i, r in enumerate(body)), sheet=name, warnings=warnings)
    df = pd.DataFrame(body, columns=columns)
    return blocks, TabularSource(name=table_name(name), path=_write_parquet(df), rows=len(df), columns=columns)


def parse_xlsx(path: str) -> ParsedDocument:
    import openpyxl

    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001
        raise InvalidDocument(f"Could not open workbook: {exc}") from exc
    blocks: list[Block] = []
    tables: list[TabularSource] = []
    warnings: list[str] = []
    for ws in wb.worksheets:
        rows = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i > MAX_ROWS_TABLE:
                warnings.append(f"{ws.title}: truncated at {MAX_ROWS_TABLE} rows.")
                break
            rows.append(list(row))
        b, t = _sheet_to_doc(ws.title, rows, warnings)
        blocks.extend(b)
        if t:
            tables.append(t)
    wb.close()
    return ParsedDocument(blocks=blocks, parser="xlsx", page_count=len(tables), tables=tables, warnings=_dedupe(warnings), metadata={"sheets": [t.name for t in tables]})


def parse_xls(path: str) -> ParsedDocument:
    import xlrd

    try:
        book = xlrd.open_workbook(path, on_demand=True)
    except Exception as exc:  # noqa: BLE001
        raise InvalidDocument(f"Could not open .xls workbook: {exc}") from exc
    blocks: list[Block] = []
    tables: list[TabularSource] = []
    warnings: list[str] = []
    for sheet in book.sheets():
        rows = [sheet.row_values(r) for r in range(min(sheet.nrows, MAX_ROWS_TABLE))]
        b, t = _sheet_to_doc(sheet.name, rows, warnings)
        blocks.extend(b)
        if t:
            tables.append(t)
    return ParsedDocument(blocks=blocks, parser="xls", page_count=len(tables), tables=tables, warnings=_dedupe(warnings))


def parse_ods(path: str) -> ParsedDocument:
    from app.ingestion.parsers.documents import _ODF_NS, _odf_content, _odf_text

    root = _odf_content(path)
    blocks: list[Block] = []
    tables: list[TabularSource] = []
    warnings: list[str] = []
    for table in root.iter(f"{{{_ODF_NS['table']}}}table"):
        name = table.get(f"{{{_ODF_NS['table']}}}name", "Sheet")
        rows: list[list[Any]] = []
        for row in table.iter(f"{{{_ODF_NS['table']}}}table-row"):
            repeat = min(int(row.get(f"{{{_ODF_NS['table']}}}number-rows-repeated", "1")), 1000)
            cells: list[Any] = []
            for cell in row.findall("table:table-cell", _ODF_NS):
                n = min(int(cell.get(f"{{{_ODF_NS['table']}}}number-columns-repeated", "1")), 200)
                cells.extend([_odf_text(cell)] * n)
            while cells and cells[-1] == "":
                cells.pop()
            if any(cells):
                rows.extend([cells] * (repeat if any(cells) else 1))
        b, t = _sheet_to_doc(name, rows, warnings)
        blocks.extend(b)
        if t:
            tables.append(t)
    return ParsedDocument(blocks=blocks, parser="ods", page_count=len(tables), tables=tables, warnings=warnings)


# ----------------------------------------------------------------------------- JSON


def _flatten(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list) and obj and not all(isinstance(x, (dict, list)) for x in obj):
        out.append((prefix, ", ".join(str(x) for x in obj[:100])))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:1000]):
            out.extend(_flatten(v, f"{prefix}[{i}]"))
    else:
        out.append((prefix, obj))
    return out


def _records_doc(records: list[dict[str, Any]], name: str, parser: str, warnings: list[str]) -> ParsedDocument:
    df = pd.json_normalize(records, max_level=2)
    df.columns = _clean_columns(df.columns)
    rows = ((i + 1, list(r)) for i, r in enumerate(df.itertuples(index=False, name=None)))
    blocks = rows_to_blocks(list(df.columns), rows, sheet=None, warnings=warnings)
    tables = [TabularSource(name=table_name(name), path=_write_parquet(df), rows=len(df), columns=list(df.columns))]
    return ParsedDocument(blocks=blocks, parser=parser, tables=tables, warnings=_dedupe(warnings), metadata={"records": len(df)})


def parse_json(path: str) -> ParsedDocument:
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        data = json.loads(raw.decode(_detect_encoding(raw), errors="replace"))
    except json.JSONDecodeError as exc:
        raise InvalidDocument(f"Invalid JSON: {exc}") from exc
    name = os.path.splitext(os.path.basename(path))[0]
    if isinstance(data, list) and data and all(isinstance(x, dict) for x in data[:100]):
        return _records_doc(data[:MAX_ROWS_TABLE], name, "json", [])
    pairs = _flatten(data)
    blocks: list[Block] = []
    for i in range(0, len(pairs), 25):
        group = pairs[i : i + 25]
        blocks.append(Block(text="\n".join(f"{k}: {v}" for k, v in group), kind="paragraph", meta={"keys": [k for k, _ in group][:5]}))
    return ParsedDocument(blocks=blocks, parser="json")


def parse_jsonl(path: str) -> ParsedDocument:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                warnings.append(f"Skipped invalid JSON on line {n}.")
                continue
            records.append(obj if isinstance(obj, dict) else {"value": obj})
            if len(records) >= MAX_ROWS_TABLE:
                break
    if not records:
        raise InvalidDocument("No JSON records found.")
    return _records_doc(records, os.path.splitext(os.path.basename(path))[0], "jsonl", warnings[:20])


# --------------------------------------------------------------------------- Parquet


def parse_parquet(path: str) -> ParsedDocument:
    import pyarrow.parquet as pq

    try:
        pf = pq.ParquetFile(path)
    except Exception as exc:  # noqa: BLE001
        raise InvalidDocument(f"Invalid Parquet file: {exc}") from exc
    columns = _clean_columns(pf.schema_arrow.names)
    warnings: list[str] = []
    blocks: list[Block] = []
    seen = 0
    for batch in pf.iter_batches(batch_size=10_000):
        if seen >= MAX_ROWS_AS_TEXT:
            break
        frame = batch.to_pandas()
        rows = ((seen + i + 1, list(r)) for i, r in enumerate(frame.itertuples(index=False, name=None)))
        blocks.extend(rows_to_blocks(columns, rows, sheet=None, warnings=warnings))
        seen += len(frame)
    name = os.path.splitext(os.path.basename(path))[0]
    rows_total = pf.metadata.num_rows
    if rows_total > MAX_ROWS_AS_TEXT:
        warnings.append(f"Only the first {MAX_ROWS_AS_TEXT} rows are indexed as text; all rows remain queryable with SQL.")
    tables = [TabularSource(name=table_name(name), path=path, rows=rows_total, columns=columns)]
    return ParsedDocument(blocks=blocks, parser="parquet", tables=tables, warnings=_dedupe(warnings), metadata={"rows": rows_total})


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
