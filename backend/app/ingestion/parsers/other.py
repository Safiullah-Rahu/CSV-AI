"""Presentations, images (OCR), source code and Jupyter notebooks."""

from __future__ import annotations

import json
import re

from app.ingestion.parsers.documents import convert_with_libreoffice, ocr_available, read_text
from app.ingestion.types import Block, InvalidDocument, ParsedDocument, UnsupportedFormat

# ----------------------------------------------------------------------- presentations


def parse_pptx(path: str) -> ParsedDocument:
    from pptx import Presentation

    try:
        prs = Presentation(path)
    except Exception as exc:  # noqa: BLE001
        raise InvalidDocument(f"Could not open presentation: {exc}") from exc
    blocks: list[Block] = []
    title = None
    for number, slide in enumerate(prs.slides, start=1):
        slide_title = None
        if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
            slide_title = slide.shapes.title.text_frame.text.strip() or None
            title = title or slide_title
        section = [slide_title] if slide_title else [f"Slide {number}"]
        for shape in slide.shapes:
            if shape.has_text_frame and shape != slide.shapes.title:
                text = "\n".join(p.text for p in shape.text_frame.paragraphs if p.text.strip()).strip()
                if text:
                    blocks.append(Block(text=text, kind="paragraph", page=number, section=section, meta={"slide": number}))
            if getattr(shape, "has_table", False) and shape.has_table:
                rows = ["| " + " | ".join(c.text.replace("|", "/").strip() for c in row.cells) + " |" for row in shape.table.rows]
                if rows:
                    blocks.append(Block(text="\n".join(rows), kind="table", page=number, section=section, meta={"slide": number}))
        if slide_title:
            blocks.insert(len(blocks), Block(text=slide_title, kind="heading", page=number, section=section, meta={"slide": number}))
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip() if slide.notes_slide.notes_text_frame else ""
            if notes:
                blocks.append(Block(text=f"Speaker notes: {notes}", kind="paragraph", page=number, section=section, meta={"slide": number, "notes": True}))
    return ParsedDocument(blocks=blocks, parser="pptx", title=title, page_count=len(prs.slides))


def parse_ppt(path: str) -> ParsedDocument:
    doc = parse_pptx(convert_with_libreoffice(path, "pptx"))
    doc.parser = "ppt(libreoffice)"
    return doc


def parse_odp(path: str) -> ParsedDocument:
    from app.ingestion.parsers.documents import _ODF_NS, _odf_content, _odf_text

    root = _odf_content(path)
    blocks: list[Block] = []
    pages = root.findall(".//draw:page", _ODF_NS)
    for number, page in enumerate(pages, start=1):
        name = page.get(f"{{{_ODF_NS['draw']}}}name") or f"Slide {number}"
        for p in page.iter(f"{{{_ODF_NS['text']}}}p"):
            text = _odf_text(p)
            if text:
                blocks.append(Block(text=text, kind="paragraph", page=number, section=[name], meta={"slide": number}))
    return ParsedDocument(blocks=blocks, parser="odp", page_count=len(pages))


# ------------------------------------------------------------------------------ images


def parse_image(path: str) -> ParsedDocument:
    if not ocr_available():
        raise UnsupportedFormat("Images need OCR: install Tesseract on the worker (pip extra 'ocr').")
    import pytesseract
    from PIL import Image

    with Image.open(path) as img:
        frames = getattr(img, "n_frames", 1)
        blocks: list[Block] = []
        for frame in range(min(frames, 200)):
            img.seek(frame)
            text = pytesseract.image_to_string(img.convert("RGB"))
            for para in re.split(r"\n\s*\n", text):
                if para.strip():
                    blocks.append(Block(text=para.strip(), kind="paragraph", page=frame + 1, meta={"ocr": True}))
        meta = {"width": img.width, "height": img.height, "format": img.format}
    return ParsedDocument(blocks=blocks, parser="image-ocr", page_count=frames, metadata=meta, warnings=[] if blocks else ["OCR found no text in the image."])


# -------------------------------------------------------------------------------- code

_DEF = re.compile(
    r"^(?:\s{0,4})(?:export\s+)?(?:async\s+)?(?:def |class |function |func |fn |pub fn |public |private |protected |interface |type |struct |impl |CREATE |create |module |package )",
)


def parse_code(path: str, ext: str) -> ParsedDocument:
    text = read_text(path)
    lines = text.splitlines()
    blocks: list[Block] = []
    start = 0
    for i in range(1, len(lines) + 1):
        boundary = i == len(lines) or (_DEF.match(lines[i]) and i - start >= 8) or i - start >= 80
        if boundary:
            chunk = "\n".join(lines[start:i]).strip("\n")
            if chunk.strip():
                blocks.append(Block(text=chunk, kind="code", meta={"line_start": start + 1, "line_end": i, "language": ext}))
            start = i
    return ParsedDocument(blocks=blocks, parser=f"code:{ext}", metadata={"lines": len(lines), "language": ext})


def parse_ipynb(path: str) -> ParsedDocument:
    try:
        nb = json.loads(read_text(path))
    except json.JSONDecodeError as exc:
        raise InvalidDocument(f"Invalid notebook: {exc}") from exc
    blocks: list[Block] = []
    trail: list[str] = []
    for index, cell in enumerate(nb.get("cells", []), start=1):
        source = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
        if not source.strip():
            continue
        if cell.get("cell_type") == "markdown":
            heading = next((ln.lstrip("#").strip() for ln in source.splitlines() if ln.startswith("#")), None)
            if heading:
                trail = [heading]
            blocks.append(Block(text=source.strip(), kind="paragraph", section=list(trail), meta={"cell": index}))
        else:
            out_text = []
            for out in cell.get("outputs", [])[:3]:
                txt = out.get("text") or (out.get("data", {}) or {}).get("text/plain")
                if txt:
                    out_text.append("".join(txt) if isinstance(txt, list) else str(txt))
            body = source.strip() + (("\n# output:\n" + "\n".join(out_text)[:2000]) if out_text else "")
            blocks.append(Block(text=body, kind="code", section=list(trail), meta={"cell": index}))
    return ParsedDocument(blocks=blocks, parser="ipynb", metadata={"cells": len(nb.get("cells", []))})
