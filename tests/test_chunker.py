from secrag.ingestion.chunker import MAX_TOKENS, approx_tokens, chunk_sections
from secrag.ingestion.parse10k import Section


def make_section(item, title, blocks):
    return Section(item=item, title=title, blocks=blocks)


LONG_PARA = "Revenue increased due to strong demand across segments. " * 20  # ~280 tokens


def test_chunks_never_cross_item_boundaries():
    sections = [
        make_section("1", "Business", [("text", LONG_PARA), ("text", LONG_PARA)]),
        make_section("1a", "Risk Factors", [("text", LONG_PARA)]),
    ]
    chunks = chunk_sections(sections, ticker="ACME", fiscal_year=2025)
    items = {c.meta["item"] for c in chunks}
    assert items == {"1", "1a"}
    for c in chunks:
        if c.meta["item"] == "1":
            assert "Item 1:" in c.content.splitlines()[0]
        else:
            assert "Item 1A:" in c.content.splitlines()[0]


def test_every_chunk_has_context_prefix():
    sections = [make_section("7", "MD&A", [("text", LONG_PARA)])]
    chunks = chunk_sections(sections, ticker="NVDA", fiscal_year=2026)
    assert all(c.content.startswith("[NVDA 10-K FY2026 — Item 7: MD&A]") for c in chunks)


def test_chunk_sizes_bounded():
    sections = [make_section("1a", "Risk Factors", [("text", LONG_PARA)] * 8)]
    chunks = chunk_sections(sections, ticker="ACME", fiscal_year=2025)
    assert len(chunks) >= 4
    assert all(c.token_count <= MAX_TOKENS + 120 for c in chunks)  # +context/join slack


def test_oversized_single_paragraph_split_on_sentences():
    huge = "This is a sentence about anvils. " * 120  # ~1000 tokens, one block
    sections = [make_section("1", "Business", [("text", huge)])]
    chunks = chunk_sections(sections, ticker="ACME", fiscal_year=2025)
    assert len(chunks) >= 2
    assert all(c.token_count <= MAX_TOKENS + 120 for c in chunks)


def test_table_is_isolated_chunk_with_flag():
    md = "| Year | Revenue |\n| --- | --- |\n| 2025 | $100 |\n| 2024 | $80 |"
    sections = [make_section("8", "Financial Statements", [("text", LONG_PARA), ("table", md)])]
    chunks = chunk_sections(sections, ticker="ACME", fiscal_year=2025)
    table_chunks = [c for c in chunks if c.meta["is_table"]]
    assert len(table_chunks) == 1
    assert "| 2025 | $100 |" in table_chunks[0].content
    assert all(not c.meta["is_table"] for c in chunks if c is not table_chunks[0])


def test_huge_table_split_repeats_header():
    rows = "\n".join(f"| item{i} | {i} |" for i in range(400))
    md = f"| Name | Value |\n| --- | --- |\n{rows}"
    sections = [make_section("8", "Financial Statements", [("table", md)])]
    chunks = chunk_sections(sections, ticker="ACME", fiscal_year=2025)
    assert len(chunks) >= 2
    for c in chunks:
        assert "| Name | Value |" in c.content  # header repeated in every part


def test_tiny_trailing_text_merges_into_previous():
    sections = [make_section("1", "Business", [("text", LONG_PARA), ("text", "Short tail.")])]
    chunks = chunk_sections(sections, ticker="ACME", fiscal_year=2025)
    assert len(chunks) == 1
    assert "Short tail." in chunks[0].content


def test_token_count_matches_content():
    sections = [make_section("1", "Business", [("text", LONG_PARA)])]
    chunks = chunk_sections(sections, ticker="ACME", fiscal_year=2025)
    for c in chunks:
        assert c.token_count == approx_tokens(c.content)
