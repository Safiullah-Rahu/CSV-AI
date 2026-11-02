"""Object storage for original files and derived artifacts.

Keys are always ``<org_id>/<area>/<uuid>/<safe-filename>`` — never user
controlled paths — so one tenant can never address another tenant's objects
and path traversal is impossible.
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Protocol

from app.config import get_settings
from app.utils.errors import NotFound

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    base = os.path.basename(name.replace("\\", "/")) or "file"
    cleaned = _SAFE.sub("_", base).strip("._") or "file"
    return cleaned[:180]


def make_key(org_id: uuid.UUID, area: str, filename: str) -> str:
    return f"{org_id}/{area}/{uuid.uuid4().hex}/{safe_filename(filename)}"


class Storage(Protocol):
    def put_file(self, key: str, src_path: str) -> int: ...
    def put_bytes(self, key: str, data: bytes) -> int: ...
    def open(self, key: str) -> BinaryIO: ...
    def local_path(self, key: str) -> Iterator[str]: ...
    def delete(self, key: str) -> None: ...
    def delete_prefix(self, prefix: str) -> int: ...
    def exists(self, key: str) -> bool: ...
    def usage_bytes(self, prefix: str) -> int: ...
    def health(self) -> bool: ...


class LocalStorage:
    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents and path != self.root:
            raise ValueError("Invalid storage key")
        return path

    def put_file(self, key: str, src_path: str) -> int:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_path, dest)
        return dest.stat().st_size

    def put_bytes(self, key: str, data: bytes) -> int:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return len(data)

    def open(self, key: str) -> BinaryIO:
        path = self._path(key)
        if not path.exists():
            raise NotFound("Stored file not found.")
        return path.open("rb")

    @contextmanager
    def local_path(self, key: str) -> Iterator[str]:
        path = self._path(key)
        if not path.exists():
            raise NotFound("Stored file not found.")
        yield str(path)

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()
            # prune the per-object directory
            try:
                path.parent.rmdir()
            except OSError:
                pass

    def delete_prefix(self, prefix: str) -> int:
        base = self._path(prefix)
        if not base.exists():
            return 0
        count = sum(1 for p in base.rglob("*") if p.is_file())
        shutil.rmtree(base, ignore_errors=True)
        return count

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def usage_bytes(self, prefix: str) -> int:
        base = self._path(prefix)
        if not base.exists():
            return 0
        return sum(p.stat().st_size for p in base.rglob("*") if p.is_file())

    def health(self) -> bool:
        probe = self.root / ".health"
        probe.write_text("ok")
        return probe.read_text() == "ok"


class S3Storage:
    """S3-compatible storage (AWS S3, MinIO). Requires ``boto3``."""

    def __init__(self) -> None:
        import boto3

        s = get_settings()
        self.bucket = s.s3_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=s.s3_endpoint_url,
            aws_access_key_id=s.s3_access_key,
            aws_secret_access_key=s.s3_secret_key,
            region_name=s.s3_region,
        )

    def put_file(self, key: str, src_path: str) -> int:
        self.client.upload_file(src_path, self.bucket, key, ExtraArgs={"ServerSideEncryption": "AES256"})
        return os.path.getsize(src_path)

    def put_bytes(self, key: str, data: bytes) -> int:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ServerSideEncryption="AES256")
        return len(data)

    def open(self, key: str) -> BinaryIO:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
        except self.client.exceptions.NoSuchKey as exc:
            raise NotFound("Stored file not found.") from exc

    @contextmanager
    def local_path(self, key: str) -> Iterator[str]:
        import tempfile

        fd, tmp = tempfile.mkstemp(suffix="_" + safe_filename(key.rsplit("/", 1)[-1]))
        os.close(fd)
        try:
            self.client.download_file(self.bucket, key, tmp)
            yield tmp
        finally:
            os.unlink(tmp)

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def delete_prefix(self, prefix: str) -> int:
        count = 0
        paginator = self.client.get_paginator("list_objects_v2")
        for pg in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            objs = [{"Key": o["Key"]} for o in pg.get("Contents", [])]
            if objs:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": objs})
                count += len(objs)
        return count

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:  # noqa: BLE001
            return False

    def usage_bytes(self, prefix: str) -> int:
        total = 0
        paginator = self.client.get_paginator("list_objects_v2")
        for pg in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            total += sum(o["Size"] for o in pg.get("Contents", []))
        return total

    def health(self) -> bool:
        self.client.head_bucket(Bucket=self.bucket)
        return True


@lru_cache(maxsize=1)
def get_storage() -> LocalStorage | S3Storage:
    settings = get_settings()
    if settings.storage_backend == "s3":
        return S3Storage()
    return LocalStorage(settings.storage_local_root)
