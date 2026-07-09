from pathlib import Path

import pytest

from secrag.ingestion.parse10k import parse_10k, parse_report

FIXTURE = Path(__file__).parent / "fixtures" / "mini_10k.html"
NVDA = Path("data/raw/NVDA/FY2026_000104581026000021.html")


@pytest.fixture(scope="module")
def sections():
    return parse_10k(FIXTURE.read_text(encoding="utf-8"))


def test_sections_in_order_with_preamble(sections):
    assert [s.item for s in sections] == [None, "1", "1a", "7", "8"]


def test_toc_table_does_not_create_sections(sections):
    # The TOC (a table before Item 1) stays in the cover section as a table block.
    cover = sections[0]
    assert cover.item is None
    assert any("ACME CORP" in b[1] for b in cover.blocks)


def test_long_cross_reference_paragraph_is_not_a_heading(sections):
    item1 = sections[1]
    assert any("anvil supply shortages" in b[1] for b in item1.blocks)


def test_out_of_order_repeat_is_body_text_not_heading(sections):
    item7 = sections[3]
    assert item7.item == "7"
    assert any(b[1] == "Item 1." for b in item7.blocks)


def test_heading_inside_layout_table_is_detected(sections):
    item8 = sections[4]
    assert item8.item == "8"
    assert any("audited statements" in b[1] for b in item8.blocks)


def test_financial_table_becomes_markdown_with_empty_columns_dropped(sections):
    tables = [b[1] for b in sections[2].blocks if b[0] == "table"]
    assert len(tables) == 1
    md = tables[0]
    assert md.splitlines()[0] == "| Fiscal year | Revenue | Net income |"
    assert "| 2025 | $ 100 | $ 25 |" in md


def test_hidden_content_excluded(sections):
    all_text = " ".join(b[1] for s in sections for b in s.blocks)
    assert "HIDDEN XBRL" not in all_text


def test_parse_report_flags_missing_core_items(sections):
    report = parse_report(sections)
    assert report["missing_core_items"] == []
    assert report["n_tables"] == 2  # the TOC table (cover) + the 1A financial table
    assert report["chars_per_item"]["1a"] > 0


@pytest.mark.integration
def test_real_nvda_filing_parses_core_items():
    if not NVDA.exists():
        pytest.skip("real NVDA filing not downloaded")
    sections = parse_10k(NVDA.read_text(encoding="utf-8"))
    report = parse_report(sections)
    assert report["missing_core_items"] == []
    found = set(report["items_found"])
    assert {"1", "1a", "7", "7a", "8"} <= found
    # Risk Factors is one of the biggest narrative sections in any 10-K.
    assert report["chars_per_item"]["1a"] > 50_000
    assert report["n_tables"] > 20
