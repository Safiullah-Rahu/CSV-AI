"""Text documents: PDF, Word, RTF, OpenDocument text, HTML, Markdown, XML, EPUB, plain text."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from app.data.loader import _detect_encoding
from app.ingestion.types import Block, InvalidDocument, ParsedDocument, UnsupportedFormat
from app.utils.logging import get_logger

log = get_logger("parsers.documents")

MAX_TEXT_BYTES = 200 * 1024 * 1024
_HEADING_MD = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def read_text(path: str) -> str:
    size = os.path.getsize(path)
    if size > MAX_TEXT_BYTES:
        raise InvalidDocument("Text file is too large to index.")
    with open(path, "rb") as fh:
        raw = fh.read()
    return raw.decode(_detect_encoding(raw), errors="replace")


# ------------------------------------------------------------------------------ plain


def parse_txt(path: str) -> ParsedDocument:
    text = read_text(path)
    blocks = [Block(text=p.strip(), kind="paragraph", meta={"para": i}) for i, p in enumerate(re.split(r"\n\s*\n", text)) if p.strip()]
    return ParsedDocument(blocks=blocks, parser="text")


def parse_markdown_text(text: str, parser: str = "markdown") -> ParsedDocument:
    blocks: list[Block] = []
    trail: list[tuple[int, str]] = []
    buf: list[str] = []
    in_code = False
    title = None

    def flush() -> None:
        content = "\n".join(buf).strip()
        if content:
            kind = "code" if content.startswith("```") else ("table" if content.lstrip().startswith("|") else "paragraph")
            blocks.append(Block(text=content, kind=kind, section=[t for _, t in trail]))
        buf.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            buf.append(line)
            if not in_code:
                flush()
            continue
        m = None if in_code else _HEADING_MD.match(line)
        if m:
            flush()
            level, heading = len(m.group(1)), m.group(2).strip()
            title = title or heading
            trail = [(lvl, t) for lvl, t in trail if lvl < level] + [(level, heading)]
            blocks.append(Block(text=heading, kind="heading", section=[t for _, t in trail], meta={"level": level}))
        elif not line.strip() and not in_code:
            flush()
        else:
            buf.append(line)
    flush()
    return ParsedDocument(blocks=blocks, parser=parser, title=title)


def parse_markdown(path: str) -> ParsedDocument:
    return parse_markdown_text(read_text(path))


def parse_rtf(path: str) -> ParsedDocument:
    from striprtf.striprtf import rtf_to_text

    text = rtf_to_text(read_text(path), errors="ignore")
    doc = parse_txt_from_string(text)
    doc.parser = "rtf"
    return doc


def parse_txt_from_string(text: str) -> ParsedDocument:
    blocks = [Block(text=p.strip(), kind="paragraph") for p in re.split(r"\n\s*\n", text) if p.strip()]
    return ParsedDocument(blocks=blocks, parser="text")


# ------------------------------------------------------------------------------- HTML


def html_to_blocks(html: str, *, url: str | None = None) -> tuple[list[Block], str | None]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "template", "iframe", "svg", "form", "nav", "footer"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else None
    body = soup.body or soup
    blocks: list[Block] = []
    trail: list[tuple[int, str]] = []
    meta_base = {"url": url} if url else {}
    for el in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table", "blockquote", "dd", "dt"]):
        if el.find_parent(["table", "pre"]) is not None and el.name not in ("table", "pre"):
            continue
        if el.name == "li" and el.find_parent("li") is not None:
            continue
        if el.name[0] == "h" and el.name[1:].isdigit():
            level = int(el.name[1])
            heading = el.get_text(" ", strip=True)
            if not heading:
                continue
            trail = [(lv, t) for lv, t in trail if lv < level] + [(level, heading)]
            blocks.append(Block(text=heading, kind="heading", section=[t for _, t in trail], meta={**meta_base, "level": level}))
        elif el.name == "table":
            rows = []
            for tr in el.find_all("tr"):
                cells = [c.get_text(" ", strip=True).replace("|", "/") for c in tr.find_all(["th", "td"])]
                if cells:
                    rows.append("| " + " | ".join(cells) + " |")
            if rows:
                if len(rows) > 1:
                    rows.insert(1, "|" + " --- |" * (rows[0].count("|") - 1))
                blocks.append(Block(text="\n".join(rows), kind="table", section=[t for _, t in trail], meta=dict(meta_base)))
        else:
            text = el.get_text(" ", strip=True)
            if text:
                kind = "code" if el.name == "pre" else ("list" if el.name == "li" else "paragraph")
                blocks.append(Block(text=("- " + text) if kind == "list" else text, kind=kind, section=[t for _, t in trail], meta=dict(meta_base)))
    # Text that sits outside block elements (e.g. bare text in <body>/<div>) is kept too.
    from bs4 import NavigableString

    captured = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table", "blockquote", "dd", "dt", "title"}
    loose: list[str] = []
    for node in body.descendants:
        if isinstance(node, NavigableString) and node.strip() and not any(getattr(a, "name", None) in captured for a in node.parents):
            loose.append(" ".join(node.split()))
    if loose:
        blocks.append(Block(text=" ".join(loose), kind="paragraph", meta=dict(meta_base)))
    return blocks, title


def parse_html(path: str) -> ParsedDocument:
    blocks, title = html_to_blocks(read_text(path))
    return ParsedDocument(blocks=blocks, parser="html", title=title)


def parse_xml(path: str) -> ParsedDocument:
    from lxml import etree

    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False, load_dtd=False)
    try:
        tree = etree.parse(path, parser)
    except etree.XMLSyntaxError as exc:
        raise InvalidDocument(f"Invalid XML: {exc}") from exc
    blocks: list[Block] = []
    for el in tree.iter():
        if not isinstance(el.tag, str):
            continue
        text = (el.text or "").strip()
        if text:
            path_str = tree.getpath(el)
            blocks.append(Block(text=f"{etree.QName(el).localname}: {text}", kind="paragraph", meta={"xpath": path_str}))
    return ParsedDocument(blocks=blocks, parser="xml")


# ------------------------------------------------------------------------------- PDF


def parse_pdf(path: str, *, extract_tables: bool = True, ocr: bool = True) -> ParsedDocument:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise InvalidDocument("PDF is password protected.") from exc
        pages = reader.pages
    except PdfReadError as exc:
        raise InvalidDocument(f"Could not read PDF: {exc}") from exc
    meta = {}
    try:
        info = reader.metadata or {}
        meta = {k.strip("/").lower(): str(v) for k, v in info.items() if v}
    except Exception:  # noqa: BLE001
        pass
    blocks: list[Block] = []
    warnings: list[str] = []
    empty_pages: list[int] = []
    for i, page in enumerate(pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Page {i}: text extraction failed ({type(exc).__name__}).")
            text = ""
        text = re.sub(r"-\n(\w)", r"\1", text)  # de-hyphenate line breaks
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paras:
            empty_pages.append(i)
        for p in paras:
            blocks.append(Block(text=p, kind="paragraph", page=i))
    if extract_tables and len(pages) <= 300:
        blocks.extend(_pdf_tables(path, warnings))
    if empty_pages:
        ocr_blocks = _pdf_ocr(path, empty_pages, warnings) if ocr else []
        blocks.extend(ocr_blocks)
        missing = sorted(set(empty_pages) - {b.page for b in ocr_blocks})
        if missing:
            warnings.append(
                f"{len(missing)} page(s) have no extractable text (scanned?): {missing[:20]}"
                + ("" if ocr_available() else " — install Tesseract OCR to index them.")
            )
    blocks.sort(key=lambda b: (b.page or 0, 0 if b.kind != "table" else 1))
    return ParsedDocument(blocks=blocks, parser="pdf", title=meta.get("title"), page_count=len(pages), metadata=meta, warnings=warnings)


def _pdf_tables(path: str, warnings: list[str]) -> list[Block]:
    try:
        import pdfplumber
    except ImportError:
        return []
    out: list[Block] = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                for t_index, table in enumerate(page.extract_tables() or []):
                    rows = [[(c or "").replace("\n", " ").replace("|", "/").strip() for c in row] for row in table if row]
                    rows = [r for r in rows if any(r)]
                    if len(rows) < 2:
                        continue
                    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * len(rows[0])]
                    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
                    out.append(Block(text="\n".join(lines), kind="table", page=i, meta={"table_index": t_index}))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Table extraction skipped ({type(exc).__name__}).")
    return out


def ocr_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


def _pdf_ocr(path: str, pages: list[int], warnings: list[str]) -> list[Block]:
    if not ocr_available():
        return []
    import pypdfium2 as pdfium
    import pytesseract

    out: list[Block] = []
    pdf = pdfium.PdfDocument(path)
    for number in pages[:500]:
        image = pdf[number - 1].render(scale=2.0).to_pil()
        text = pytesseract.image_to_string(image)
        for p in re.split(r"\n\s*\n", text):
            if p.strip():
                out.append(Block(text=p.strip(), kind="paragraph", page=number, meta={"ocr": True}))
    if out:
        warnings.append(f"OCR applied to {len({b.page for b in out})} page(s).")
    return out


# --------------------------------------------------------------------------- Word


def parse_docx(path: str) -> ParsedDocument:
    import docx
    from docx.document import Document as _Doc
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document: _Doc = docx.Document(path)
    except Exception as exc:  # noqa: BLE001
        raise InvalidDocument(f"Could not open Word document: {exc}") from exc
    blocks: list[Block] = []
    trail: list[tuple[int, str]] = []
    title = None
    body = document.element.body
    t_index = 0
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, document)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name or "").lower() if para.style is not None else ""
            level = None
            if style.startswith("heading"):
                digits = "".join(ch for ch in style if ch.isdigit())
                level = int(digits) if digits else 1
            elif style == "title":
                level = 1
            if level:
                title = title or text
                trail = [(lv, t) for lv, t in trail if lv < level] + [(level, text)]
                blocks.append(Block(text=text, kind="heading", section=[t for _, t in trail], meta={"level": level}))
            else:
                kind = "list" if "list" in style else "paragraph"
                blocks.append(Block(text=text, kind=kind, section=[t for _, t in trail]))
        elif tag == "tbl":
            table = Table(child, document)
            rows = []
            for row in table.rows:
                cells = [c.text.replace("\n", " ").replace("|", "/").strip() for c in row.cells]
                rows.append("| " + " | ".join(cells) + " |")
            if rows:
                if len(rows) > 1:
                    rows.insert(1, "|" + " --- |" * (rows[0].count("|") - 1))
                blocks.append(Block(text="\n".join(rows), kind="table", section=[t for _, t in trail], meta={"table_index": t_index}))
                t_index += 1
    props = document.core_properties
    meta = {k: str(v) for k, v in {"author": props.author, "title": props.title, "created": props.created, "modified": props.modified}.items() if v}
    return ParsedDocument(blocks=blocks, parser="docx", title=props.title or title, metadata=meta)


def libreoffice_binary() -> str | None:
    return shutil.which("soffice") or shutil.which("libreoffice")


def convert_with_libreoffice(path: str, target_ext: str) -> str:
    """Convert legacy Office formats (doc/ppt/xls) with a headless LibreOffice, if installed."""
    binary = libreoffice_binary()
    if not binary:
        raise UnsupportedFormat(
            "Legacy Office formats (.doc/.ppt) need LibreOffice installed on the worker; "
            "save the file as .docx/.pptx or install LibreOffice."
        )
    out_dir = tempfile.mkdtemp(prefix="convert_")
    subprocess.run(
        [binary, "--headless", "--convert-to", target_ext, "--outdir", out_dir, path],
        check=True,
        timeout=180,
        capture_output=True,
    )
    converted = list(Path(out_dir).glob(f"*.{target_ext}"))
    if not converted:
        raise InvalidDocument("LibreOffice conversion produced no output.")
    return str(converted[0])


def parse_doc(path: str) -> ParsedDocument:
    doc = parse_docx(convert_with_libreoffice(path, "docx"))
    doc.parser = "doc(libreoffice)"
    return doc


# ----------------------------------------------------------------------- OpenDocument


_ODF_NS = {
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
}


def _odf_content(path: str):  # noqa: ANN202
    from lxml import etree

    with zipfile.ZipFile(path) as zf:
        try:
            raw = zf.read("content.xml")
        except KeyError as exc:
            raise InvalidDocument("Not an OpenDocument file (content.xml missing).") from exc
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    return etree.fromstring(raw, parser)


def _odf_text(el) -> str:  # noqa: ANN001
    return " ".join("".join(el.itertext()).split())


def parse_odt(path: str) -> ParsedDocument:
    root = _odf_content(path)
    blocks: list[Block] = []
    trail: list[tuple[int, str]] = []
    t_index = 0
    body = root.find(".//office:text", _ODF_NS)
    if body is None:
        return ParsedDocument(blocks=[], parser="odt")
    for el in body.iter(f"{{{_ODF_NS['text']}}}h", f"{{{_ODF_NS['text']}}}p", f"{{{_ODF_NS['table']}}}table"):
        local = el.tag.rsplit("}", 1)[1]
        if local == "p" and el.getparent() is not None and el.getparent().tag.endswith("table-cell"):
            continue
        if local == "h":
            level = int(el.get(f"{{{_ODF_NS['text']}}}outline-level", "1"))
            text = _odf_text(el)
            if text:
                trail = [(lv, t) for lv, t in trail if lv < level] + [(level, text)]
                blocks.append(Block(text=text, kind="heading", section=[t for _, t in trail], meta={"level": level}))
        elif local == "p":
            text = _odf_text(el)
            if text:
                blocks.append(Block(text=text, kind="paragraph", section=[t for _, t in trail]))
        else:
            rows = ["| " + " | ".join(_odf_text(c) for c in row.findall("table:table-cell", _ODF_NS)) + " |" for row in el.iter(f"{{{_ODF_NS['table']}}}table-row")]
            if rows:
                blocks.append(Block(text="\n".join(rows), kind="table", section=[t for _, t in trail], meta={"table_index": t_index}))
                t_index += 1
    return ParsedDocument(blocks=blocks, parser="odt")


# -------------------------------------------------------------------------------- EPUB


def parse_epub(path: str) -> ParsedDocument:
    from lxml import etree

    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    blocks: list[Block] = []
    title = None
    with zipfile.ZipFile(path) as zf:
        try:
            container = etree.fromstring(zf.read("META-INF/container.xml"), parser)
        except KeyError as exc:
            raise InvalidDocument("Invalid EPUB (container.xml missing).") from exc
        opf_path = container.find(".//{*}rootfile").get("full-path")
        opf = etree.fromstring(zf.read(opf_path), parser)
        base = os.path.dirname(opf_path)
        t = opf.find(".//{*}title")
        title = t.text if t is not None else None
        manifest = {item.get("id"): item.get("href") for item in opf.iter("{*}item")}
        spine = [ref.get("idref") for ref in opf.iter("{*}itemref")]
        for chapter, idref in enumerate(spine, start=1):
            href = manifest.get(idref)
            if not href:
                continue
            name = os.path.normpath(os.path.join(base, href)).replace("\\", "/")
            try:
                html = zf.read(name).decode("utf-8", errors="replace")
            except KeyError:
                continue
            chapter_blocks, chapter_title = html_to_blocks(html)
            for b in chapter_blocks:
                b.meta["chapter"] = chapter
                b.page = chapter
                if chapter_title and not b.section:
                    b.section = [chapter_title]
            blocks.extend(chapter_blocks)
    return ParsedDocument(blocks=blocks, parser="epub", title=title, page_count=len(spine))
