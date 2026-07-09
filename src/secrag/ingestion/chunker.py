"""Section-aware chunking: pack paragraphs, isolate tables, never cross Items.

Every chunk's content starts with a context line
`[TICKER 10-K FYxxxx — Item N: Title]` — it grounds the embedding, feeds
useful terms to full-text search, and makes citations self-describing.
Token counts are the chars/4 approximation (bge tokenizer truncates at 512
real tokens; the reranker and generator see full text regardless).
"""

import re
from dataclasses import dataclass

from secrag.ingestion.parse10k import Section

TARGET_TOKENS = 400
MAX_TOKENS = 500
MIN_MERGE_TOKENS = 80  # text pieces smaller than this merge into the previous chunk

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkDraft:
    content: str
    token_count: int
    meta: dict


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _split_long_text(text: str, target: int, max_: int) -> list[str]:
    if approx_tokens(text) <= max_:
        return [text]
    parts: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        st = approx_tokens(sentence)
        if buf and buf_tokens + st > target:
            parts.append(" ".join(buf))
            buf, buf_tokens = [], 0
        buf.append(sentence)
        buf_tokens += st
    if buf:
        parts.append(" ".join(buf))
    return parts


def _split_table(markdown: str, max_: int) -> list[str]:
    if approx_tokens(markdown) <= max_:
        return [markdown]
    lines = markdown.splitlines()
    header, body = lines[:2], lines[2:]
    header_tokens = approx_tokens("\n".join(header))
    parts: list[str] = []
    buf: list[str] = []
    buf_tokens = header_tokens
    for row in body:
        rt = approx_tokens(row)
        if buf and buf_tokens + rt > max_:
            parts.append("\n".join(header + buf))
            buf, buf_tokens = [], header_tokens
        buf.append(row)
        buf_tokens += rt
    if buf:
        parts.append("\n".join(header + buf))
    return parts


def _pack_section(section: Section, target: int, max_: int) -> list[tuple[bool, str]]:
    """Return (is_table, body) proto-chunks for one section."""
    protos: list[tuple[bool, str]] = []
    buf: list[str] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf, buf_tokens
        if buf:
            protos.append((False, "\n\n".join(buf)))
            buf, buf_tokens = [], 0

    for kind, text in section.blocks:
        if kind == "table":
            flush()
            protos.extend((True, part) for part in _split_table(text, max_))
            continue
        for part in _split_long_text(text, target, max_):
            pt = approx_tokens(part)
            if buf and buf_tokens + pt > target:
                flush()
            buf.append(part)
            buf_tokens += pt
    flush()

    merged: list[tuple[bool, str]] = []
    for is_table, body in protos:
        if (
            merged
            and not is_table
            and not merged[-1][0]
            and approx_tokens(body) < MIN_MERGE_TOKENS
            and approx_tokens(merged[-1][1]) + approx_tokens(body) <= max_ + 100
        ):
            merged[-1] = (False, merged[-1][1] + "\n\n" + body)
        else:
            merged.append((is_table, body))
    return merged


def context_line(ticker: str, fiscal_year: int, section: Section) -> str:
    if section.item is None:
        return f"[{ticker} 10-K FY{fiscal_year} — Cover]"
    return f"[{ticker} 10-K FY{fiscal_year} — Item {section.item.upper()}: {section.title}]"


def chunk_sections(
    sections: list[Section],
    *,
    ticker: str,
    fiscal_year: int,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
) -> list[ChunkDraft]:
    chunks: list[ChunkDraft] = []
    for section in sections:
        ctx = context_line(ticker, fiscal_year, section)
        for is_table, body in _pack_section(section, target_tokens, max_tokens):
            content = f"{ctx}\n{body}"
            chunks.append(
                ChunkDraft(
                    content=content,
                    token_count=approx_tokens(content),
                    meta={
                        "item": section.item,
                        "item_title": section.title,
                        "is_table": is_table,
                    },
                )
            )
    return chunks
