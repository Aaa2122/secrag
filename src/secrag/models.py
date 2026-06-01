from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Computed, Date, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("ticker", "filing_type", "fiscal_year"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    ticker: Mapped[str] = mapped_column(Text)
    company_name: Mapped[str] = mapped_column(Text)
    cik: Mapped[str] = mapped_column(Text)
    filing_type: Mapped[str] = mapped_column(Text, default="10-K")
    fiscal_year: Mapped[int] = mapped_column(Integer)
    accession_number: Mapped[str] = mapped_column(Text, unique=True)
    source_url: Mapped[str] = mapped_column(Text)
    filed_at: Mapped[date | None] = mapped_column(Date)
    ingested_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int | None] = mapped_column(Integer)
    embedding = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    tsv = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
        nullable=True,
    )
    meta: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, server_default="{}")

    document: Mapped[Document] = relationship(back_populates="chunks")
