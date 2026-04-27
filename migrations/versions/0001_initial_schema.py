"""initial schema: documents and chunks with vector, tsvector, jsonb

Revision ID: 0001
Revises:
Create Date: 2026-07-07

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, TSVECTOR

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 384  # BAAI/bge-small-en-v1.5; changing the embedder means a new migration


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("company_name", sa.Text(), nullable=False),
        sa.Column("cik", sa.Text(), nullable=False),
        sa.Column("filing_type", sa.Text(), nullable=False, server_default="10-K"),
        sa.Column("fiscal_year", sa.Integer(), nullable=False),
        sa.Column("accession_number", sa.Text(), nullable=False, unique=True),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("filed_at", sa.Date(), nullable=True),
        sa.Column("ingested_at", TIMESTAMP(timezone=True), nullable=True),
        sa.UniqueConstraint("ticker", "filing_type", "fiscal_year"),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "document_id",
            sa.BigInteger(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column(
            "tsv",
            TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=True,
        ),
        sa.Column("metadata", JSONB(), nullable=False, server_default="{}"),
        sa.UniqueConstraint("document_id", "chunk_index"),
    )

    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING gin (tsv)")
    op.execute("CREATE INDEX ix_chunks_metadata ON chunks USING gin (metadata)")


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("documents")
