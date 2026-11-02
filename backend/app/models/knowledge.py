"""Knowledge bases, documents, versions, chunks, embeddings, datasets."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TenantScoped, Timestamps, UUIDPk


class KnowledgeBase(UUIDPk, TenantScoped, Timestamps, Base):
    __tablename__ = "knowledge_bases"
    __table_args__ = (UniqueConstraint("org_id", "name"),)

    name: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="SET NULL"), index=True
    )
    department_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("departments.id", ondelete="SET NULL")
    )
    owner_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, nullable=False)
    category: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    # restricted: explicit grants only | workspace: workspace members read | org: all members read
    visibility: Mapped[str] = mapped_column(String(16), default="restricted", nullable=False)
    embedding_config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    chunking_config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    retrieval_config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    generation_config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    # active | archived
    status: Mapped[str] = mapped_column(String(16), default="active", nullable=False)
    config_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_sample: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class KBConfigRevision(UUIDPk, TenantScoped, Base):
    """Version history of a knowledge base's configuration."""

    __tablename__ = "kb_config_revisions"

    kb_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    changed_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Document(UUIDPk, TenantScoped, Timestamps, Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_org_sha", "org_id", "sha256"),
        Index(
            "uq_documents_source_ref",
            "kb_id",
            "source_type",
            "source_id",
            "source_ref",
            unique=True,
            postgresql_where=text("source_ref IS NOT NULL"),
        ),
    )

    kb_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    ext: Mapped[str] = mapped_column(String(16), default="", nullable=False)
    mime_type: Mapped[str] = mapped_column(String(160), default="", nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    storage_key: Mapped[str | None] = mapped_column(String(600))
    owner_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    # queued | processing | ready | failed | deleting
    status: Mapped[str] = mapped_column(String(16), default="queued", nullable=False, index=True)
    parser: Mapped[str] = mapped_column(String(60), default="", nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    language: Mapped[str | None] = mapped_column(String(16))
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    current_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    last_indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    embedding_status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    # upload | database | integration | sample
    source_type: Mapped[str] = mapped_column(String(16), default="upload", nullable=False)
    source_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    source_ref: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    # inherit: KB permissions apply | restricted: explicit document grants also required
    acl_mode: Mapped[str] = mapped_column(String(16), default="inherit", nullable=False)


class DocumentVersion(UUIDPk, TenantScoped, Base):
    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("document_id", "version"),)

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    storage_key: Mapped[str | None] = mapped_column(String(600))
    filename: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    note: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Chunk(UUIDPk, TenantScoped, Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        Index("ix_chunks_kb_retrievable", "kb_id", "is_retrievable"),
    )

    kb_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE")
    )
    # Parents in parent-child retrieval are context-only (not matched directly).
    is_retrievable: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('simple'::regconfig, content)", persisted=True)
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChunkEmbedding(TenantScoped, Base):
    """Embedding of one chunk. Undimensioned ``vector`` + per-dimension HNSW
    partial indexes let each knowledge base choose its own model."""

    __tablename__ = "chunk_embeddings"
    __table_args__ = (Index("ix_chunk_embeddings_kb_dim", "kb_id", "dim"),)

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    kb_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE")
    )
    model: Mapped[str] = mapped_column(String(160), nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[Any] = mapped_column(Vector(), nullable=False)


class Dataset(UUIDPk, TenantScoped, Timestamps, Base):
    """A tabular document exposed for structured analysis (CSV-AI heritage)."""

    __tablename__ = "datasets"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), unique=True
    )
    kb_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    table_name: Mapped[str] = mapped_column(String(120), nullable=False)
    row_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    column_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    profile: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    # Normalised Parquet copy used by the SQL engine.
    storage_key: Mapped[str | None] = mapped_column(String(600))
    sheets: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, nullable=False)
