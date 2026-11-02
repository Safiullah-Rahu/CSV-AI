"""File type detection and upload validation.

The extension selects the parser; magic bytes must agree with it. Executables,
secrets files and archive bombs are rejected before any parser runs.
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass

import filetype

from app.ingestion.types import InvalidDocument, UnsupportedFormat

DOCUMENT_EXT = {"pdf", "doc", "docx", "txt", "rtf", "odt", "html", "htm", "md", "markdown", "xml", "epub"}
TABULAR_EXT = {"csv", "tsv", "xls", "xlsx", "xlsm", "ods", "json", "jsonl", "ndjson", "parquet"}
SLIDES_EXT = {"ppt", "pptx", "odp"}
IMAGE_EXT = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}
CODE_EXT = {
    "sql", "yaml", "yml", "log", "py", "js", "mjs", "cjs", "ts", "tsx", "jsx", "java", "c", "h", "cc", "cpp",
    "cxx", "hpp", "cs", "go", "rs", "rb", "php", "swift", "kt", "kts", "scala", "sh", "bash", "ps1", "r",
    "toml", "ini", "cfg", "conf", "properties", "gradle", "tf", "proto", "graphql", "css", "scss", "vue",
    "ipynb", "dockerfile", "makefile",
}
SUPPORTED_EXT = DOCUMENT_EXT | TABULAR_EXT | SLIDES_EXT | IMAGE_EXT | CODE_EXT

# Never ingest: credentials and private keys commonly committed by mistake.
BLOCKED_NAMES = {".env", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", ".npmrc", ".pypirc", ".netrc", "credentials"}
BLOCKED_EXT = {"pem", "key", "p12", "pfx", "keystore", "jks", "kdbx", "env", "exe", "dll", "so", "dylib", "bat", "cmd", "msi", "scr", "com", "vbs", "jar"}

# Expected magic-byte MIME per extension (when filetype can identify the content).
_MAGIC = {
    "pdf": {"application/pdf"},
    "png": {"image/png"},
    "jpg": {"image/jpeg"},
    "jpeg": {"image/jpeg"},
    "tif": {"image/tiff"},
    "tiff": {"image/tiff"},
    "bmp": {"image/bmp"},
    "webp": {"image/webp"},
    "epub": {"application/epub+zip", "application/zip"},
    "rtf": {"application/rtf", "text/rtf"},
}
_ZIP_BASED = {"docx", "xlsx", "xlsm", "pptx", "odt", "ods", "odp", "epub"}
_OLE_BASED = {"doc", "xls", "ppt"}
_EXECUTABLE_MIME = {"application/x-msdownload", "application/x-executable", "application/x-mach-binary", "application/x-elf", "application/x-sharedlib"}

MIME_BY_EXT = {
    "pdf": "application/pdf", "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain", "rtf": "application/rtf", "odt": "application/vnd.oasis.opendocument.text",
    "html": "text/html", "htm": "text/html", "md": "text/markdown", "markdown": "text/markdown",
    "xml": "application/xml", "epub": "application/epub+zip", "csv": "text/csv",
    "tsv": "text/tab-separated-values", "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12", "ods": "application/vnd.oasis.opendocument.spreadsheet",
    "json": "application/json", "jsonl": "application/x-ndjson", "ndjson": "application/x-ndjson",
    "parquet": "application/vnd.apache.parquet", "ppt": "application/vnd.ms-powerpoint",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "odp": "application/vnd.oasis.opendocument.presentation", "png": "image/png", "jpg": "image/jpeg",
    "jpeg": "image/jpeg", "tif": "image/tiff", "tiff": "image/tiff", "bmp": "image/bmp", "webp": "image/webp",
    "ipynb": "application/x-ipynb+json",
}

MAX_UNCOMPRESSED = 1024 * 1024 * 1024  # 1 GiB expanded
MAX_RATIO = 200


@dataclass
class Detected:
    ext: str
    mime: str
    family: str  # document | tabular | slides | image | code


def extension_of(filename: str) -> str:
    base = os.path.basename(filename).lower()
    if base in ("dockerfile", "makefile"):
        return base
    if "." not in base:
        return ""
    return base.rsplit(".", 1)[1]


def family_of(ext: str) -> str:
    if ext in TABULAR_EXT:
        return "tabular"
    if ext in SLIDES_EXT:
        return "slides"
    if ext in IMAGE_EXT:
        return "image"
    if ext in CODE_EXT:
        return "code"
    return "document"


def check_name(filename: str) -> str:
    base = os.path.basename(filename).lower()
    ext = extension_of(filename)
    if base in BLOCKED_NAMES or base.startswith(".env") or ext in BLOCKED_EXT:
        raise InvalidDocument(f"'{filename}' looks like a secret, key or executable file and is not accepted.")
    if ext not in SUPPORTED_EXT:
        raise UnsupportedFormat(f"File type '.{ext or '?'}' is not supported.")
    return ext


def _check_zip(path: str, ext: str) -> None:
    if not zipfile.is_zipfile(path):
        raise InvalidDocument(f"File is not a valid .{ext} archive.")
    compressed = os.path.getsize(path) or 1
    total = 0
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            total += info.file_size
            if info.file_size > 0 and info.compress_size > 0 and info.file_size / info.compress_size > MAX_RATIO * 5:
                raise InvalidDocument("Archive entry compression ratio is suspicious (possible zip bomb).")
    if total > MAX_UNCOMPRESSED or total / compressed > MAX_RATIO:
        raise InvalidDocument("Archive expands to an unsafe size (possible zip bomb).")


def detect(path: str, filename: str) -> Detected:
    ext = check_name(filename)
    size = os.path.getsize(path)
    if size == 0:
        raise InvalidDocument("File is empty.")
    kind = filetype.guess(path)
    sniffed = kind.mime if kind else None
    if sniffed in _EXECUTABLE_MIME:
        raise InvalidDocument("Executable content is not accepted.")
    expected = _MAGIC.get(ext)
    if expected and sniffed and sniffed not in expected:
        raise InvalidDocument(f"Content ({sniffed}) does not match the .{ext} extension.")
    if expected and sniffed is None and ext in ("pdf", "png", "jpg", "jpeg"):
        raise InvalidDocument(f"Content does not look like a .{ext} file.")
    if ext in _ZIP_BASED:
        _check_zip(path, ext)
    if ext in _OLE_BASED:
        with open(path, "rb") as fh:
            if fh.read(8) != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
                raise InvalidDocument(f"File is not a valid legacy .{ext} document.")
    if ext == "parquet":
        with open(path, "rb") as fh:
            if fh.read(4) != b"PAR1":
                raise InvalidDocument("File is not a valid Parquet file.")
    if family_of(ext) in ("code", "document", "tabular") and ext not in _ZIP_BASED | _OLE_BASED | {"pdf", "parquet", "epub", "rtf"}:
        with open(path, "rb") as fh:
            head = fh.read(8192)
        if b"\x00" in head and not head.startswith((b"\xff\xfe", b"\xfe\xff")):
            raise InvalidDocument("Binary content found in a text file.")
    return Detected(ext=ext, mime=MIME_BY_EXT.get(ext, sniffed or "text/plain"), family=family_of(ext))
