"""Generate small, realistic sample files in every supported format (no network)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

POLICY_PARAGRAPHS = [
    ("Annual leave", "Full-time employees accrue 25 days of paid annual leave per calendar year. Leave requests must be submitted at least two weeks in advance through the HR portal."),
    ("Sick leave", "Employees may take up to 10 days of paid sick leave per year. A medical certificate is required for absences longer than three consecutive days."),
    ("Remote work", "Staff may work remotely up to three days per week with manager approval. Core collaboration hours are 10:00 to 15:00 local time."),
]


def _pdf_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(path: Path, pages: list[list[str]]) -> Path:
    """Minimal valid PDF with Helvetica text, one content stream per page."""
    objects: list[bytes] = []
    kids = []
    font_id = 3 + 2 * len(pages)
    for i, lines in enumerate(pages):
        page_id, content_id = 3 + 2 * i, 4 + 2 * i
        kids.append(f"{page_id} 0 R")
        ops = ["BT", "/F1 11 Tf", "50 760 Td", "14 TL"]
        for line in lines:
            ops.append(f"({_pdf_escape(line)}) Tj T*")
            ops.append("T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objects.append(f"{page_id} 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >> endobj\n".encode())
        objects.append(f"{content_id} 0 obj << /Length {len(stream)} >> stream\n".encode() + stream + b"\nendstream endobj\n")
    header = b"%PDF-1.4\n"
    body = [b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n", f"2 0 obj << /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >> endobj\n".encode()]
    body += objects
    body.append(f"{font_id} 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n".encode())
    out = bytearray(header)
    offsets = []
    for obj in body:
        offsets.append(len(out))
        out += obj
    xref = len(out)
    out += f"xref\n0 {len(body) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer << /Size {len(body) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


def make_docx(path: Path) -> Path:
    import docx

    d = docx.Document()
    d.add_heading("Employee Handbook", level=1)
    for heading, text in POLICY_PARAGRAPHS:
        d.add_heading(heading, level=2)
        d.add_paragraph(text)
    t = d.add_table(rows=3, cols=2)
    for r, (a, b) in enumerate([("Grade", "Leave days"), ("Junior", "25"), ("Senior", "28")]):
        t.cell(r, 0).text, t.cell(r, 1).text = a, b
    d.save(path)
    return path


def make_xlsx(path: Path) -> Path:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Salaries"
    ws.append(["employee", "department", "salary", "start_date"])
    for row in [("Alice Moreau", "Finance", 72000, "2021-03-01"), ("Bob Tanaka", "Engineering", 98000, "2019-07-15"), ("Chen Wei", "HR", 61000, "2022-01-10")]:
        ws.append(row)
    ws2 = wb.create_sheet("Bands")
    ws2.append(["band", "min", "max"])
    ws2.append(["B1", 50000, 70000])
    ws2.append(["B2", 70000, 95000])
    wb.save(path)
    return path


def make_pptx(path: Path) -> Path:
    from pptx import Presentation

    prs = Presentation()
    for title, body in [("Quarterly results", "Revenue grew 14 percent quarter over quarter."), ("Hiring plan", "We will hire 12 engineers in Q3.")]:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    prs.save(path)
    return path


def _odf(path: Path, mimetype: str, content: str) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", mimetype, compress_type=zipfile.ZIP_STORED)
        zf.writestr("content.xml", content)
        zf.writestr("META-INF/manifest.xml", '<?xml version="1.0"?><manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"/>')
    return path


ODF_HEAD = '<?xml version="1.0" encoding="UTF-8"?><office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"><office:body>'


def make_odt(path: Path) -> Path:
    body = "".join(f'<text:h text:outline-level="2">{h}</text:h><text:p>{t}</text:p>' for h, t in POLICY_PARAGRAPHS)
    return _odf(path, "application/vnd.oasis.opendocument.text", ODF_HEAD + f"<office:text>{body}</office:text></office:body></office:document-content>")


def make_ods(path: Path) -> Path:
    rows = [("region", "revenue"), ("EMEA", "1200"), ("APAC", "950")]
    xml = "".join("<table:table-row>" + "".join(f"<table:table-cell><text:p>{c}</text:p></table:table-cell>" for c in r) + "</table:table-row>" for r in rows)
    return _odf(path, "application/vnd.oasis.opendocument.spreadsheet", ODF_HEAD + f'<office:spreadsheet><table:table table:name="Revenue">{xml}</table:table></office:spreadsheet></office:body></office:document-content>')


def make_odp(path: Path) -> Path:
    return _odf(path, "application/vnd.oasis.opendocument.presentation", ODF_HEAD + '<office:presentation><draw:page draw:name="Roadmap"><draw:frame><draw:text-box><text:p>Launch the partner portal in October.</text:p></draw:text-box></draw:frame></draw:page></office:presentation></office:body></office:document-content>')


def make_epub(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
        zf.writestr("OEBPS/content.opf", '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Security Guide</dc:title></metadata><manifest><item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="c1"/></spine></package>')
        zf.writestr("OEBPS/c1.xhtml", "<html><body><h1>Passwords</h1><p>Rotate service account passwords every 90 days.</p></body></html>")
    return path


def make_all(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    files["pdf"] = make_pdf(root / "policy.pdf", [[f"{h}: {t[:80]}", t[80:]] for h, t in POLICY_PARAGRAPHS])
    files["docx"] = make_docx(root / "handbook.docx")
    files["xlsx"] = make_xlsx(root / "salaries.xlsx")
    files["pptx"] = make_pptx(root / "results.pptx")
    files["odt"] = make_odt(root / "policy.odt")
    files["ods"] = make_ods(root / "revenue.ods")
    files["odp"] = make_odp(root / "roadmap.odp")
    files["epub"] = make_epub(root / "security.epub")
    (root / "notes.md").write_text("# Onboarding\n\n## Laptops\n\nNew hires receive a laptop on day one.\n\n## Accounts\n\nIT creates accounts within 24 hours.\n", encoding="utf-8")
    files["md"] = root / "notes.md"
    (root / "page.html").write_text("<html><head><title>Expenses</title><script>alert(1)</script></head><body><h1>Expense policy</h1><p>Meals are reimbursed up to 50 EUR per day.</p><table><tr><th>Item</th><th>Limit</th></tr><tr><td>Hotel</td><td>180</td></tr></table></body></html>", encoding="utf-8")
    files["html"] = root / "page.html"
    (root / "readme.txt").write_text("Visitor badges are issued at reception.\n\nBadges must be returned at the end of the day.", encoding="utf-8")
    files["txt"] = root / "readme.txt"
    (root / "memo.rtf").write_text(r"{\rtf1\ansi{\fonttbl\f0\fswiss Helvetica;}\f0\pard Travel must be booked through the approved agency.\par}", encoding="latin-1")
    files["rtf"] = root / "memo.rtf"
    (root / "config.xml").write_text('<?xml version="1.0"?><settings><retention days="30">Logs are kept for thirty days.</retention></settings>', encoding="utf-8")
    files["xml"] = root / "config.xml"
    (root / "orders.csv").write_text("order_id,customer,amount,month\n1,Acme,1200.50,2026-01\n2,Globex,980.00,2026-01\n3,Acme,450.25,2026-02\n4,Initech,2300.00,2026-03\n", encoding="utf-8")
    files["csv"] = root / "orders.csv"
    (root / "orders.tsv").write_text("sku\tname\tstock\nA1\tWidget\t40\nB2\tGadget\t7\n", encoding="utf-8")
    files["tsv"] = root / "orders.tsv"
    (root / "people.json").write_text(json.dumps([{"name": "Ana", "role": "Analyst", "office": {"city": "Lisbon"}}, {"name": "Raj", "role": "Engineer", "office": {"city": "Pune"}}]), encoding="utf-8")
    files["json"] = root / "people.json"
    (root / "events.jsonl").write_text('{"event": "login", "user": "ana"}\n{"event": "export", "user": "raj"}\n', encoding="utf-8")
    files["jsonl"] = root / "events.jsonl"
    import pandas as pd

    pd.DataFrame({"city": ["Lisbon", "Pune"], "employees": [120, 340]}).to_parquet(root / "offices.parquet", index=False)
    files["parquet"] = root / "offices.parquet"
    (root / "etl.py").write_text("import os\n\n\ndef load(path):\n    '''Load the nightly export.'''\n    return open(path).read()\n\n\nclass Job:\n    def run(self):\n        return load(os.environ['EXPORT'])\n", encoding="utf-8")
    files["py"] = root / "etl.py"
    (root / "schema.sql").write_text("CREATE TABLE invoices (id int primary key, total numeric);\n", encoding="utf-8")
    files["sql"] = root / "schema.sql"
    (root / "deploy.yaml").write_text("replicas: 3\nimage: api:1.4\n", encoding="utf-8")
    files["yaml"] = root / "deploy.yaml"
    nb = {"cells": [{"cell_type": "markdown", "source": ["# Churn analysis\n", "Monthly churn is 2.1 percent."]}, {"cell_type": "code", "source": ["print(2+2)"], "outputs": [{"text": ["4\n"]}]}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    (root / "churn.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    files["ipynb"] = root / "churn.ipynb"
    return files
