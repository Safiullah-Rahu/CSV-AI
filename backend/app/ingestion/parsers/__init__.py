"""Parser registry: extension → parser function."""

from __future__ import annotations

from collections.abc import Callable

from app.ingestion.parsers import documents as d
from app.ingestion.parsers import other as o
from app.ingestion.parsers import tabular as t
from app.ingestion.types import ParsedDocument, UnsupportedFormat

PARSERS: dict[str, Callable[[str], ParsedDocument]] = {
    "pdf": d.parse_pdf,
    "docx": d.parse_docx,
    "doc": d.parse_doc,
    "txt": d.parse_txt,
    "rtf": d.parse_rtf,
    "odt": d.parse_odt,
    "html": d.parse_html,
    "htm": d.parse_html,
    "md": d.parse_markdown,
    "markdown": d.parse_markdown,
    "xml": d.parse_xml,
    "epub": d.parse_epub,
    "csv": lambda p: t.parse_delimited(p, "csv"),
    "tsv": lambda p: t.parse_delimited(p, "tsv"),
    "xlsx": t.parse_xlsx,
    "xlsm": t.parse_xlsx,
    "xls": t.parse_xls,
    "ods": t.parse_ods,
    "json": t.parse_json,
    "jsonl": t.parse_jsonl,
    "ndjson": t.parse_jsonl,
    "parquet": t.parse_parquet,
    "pptx": o.parse_pptx,
    "ppt": o.parse_ppt,
    "odp": o.parse_odp,
    "ipynb": o.parse_ipynb,
}
for _img in ("png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"):
    PARSERS[_img] = o.parse_image


def parse(path: str, ext: str) -> ParsedDocument:
    from app.ingestion.detect import CODE_EXT

    fn = PARSERS.get(ext)
    if fn is not None:
        return fn(path)
    if ext in CODE_EXT:
        return o.parse_code(path, ext)
    raise UnsupportedFormat(f"No parser for .{ext}")


def format_support() -> list[dict[str, object]]:
    """Capability matrix shown in the UI (formats needing optional tools are flagged)."""
    from app.ingestion.detect import CODE_EXT, DOCUMENT_EXT, IMAGE_EXT, SLIDES_EXT, TABULAR_EXT

    ocr = d.ocr_available()
    office = d.libreoffice_binary() is not None
    rows = []
    for family, exts in (("document", DOCUMENT_EXT), ("tabular", TABULAR_EXT), ("slides", SLIDES_EXT), ("image", IMAGE_EXT), ("code", CODE_EXT)):
        for ext in sorted(exts):
            available, needs = True, None
            if ext in IMAGE_EXT:
                available, needs = ocr, "Tesseract OCR"
            elif ext in ("doc", "ppt"):
                available, needs = office, "LibreOffice"
            rows.append({"ext": ext, "family": family, "available": available, "requires": needs})
    rows.append({"ext": "pdf (scanned pages)", "family": "document", "available": ocr, "requires": "Tesseract OCR"})
    return rows
