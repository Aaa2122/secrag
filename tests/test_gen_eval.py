from secrag.evals.gen import (
    extract_numbers,
    fabricated_numbers,
    is_refusal,
    key_figure_present,
    parse_citations,
)


def test_extract_numbers_normalizes_and_ignores_citations():
    assert extract_numbers("repurchased 282 million shares for $40.4 billion [2]") == {
        "282",
        "40.4",
    }
    assert extract_numbers("equity was $ 56,950 in FY2025") == {"56950", "2025"}
    assert extract_numbers("grew 162%") == {"162"}


def test_fabricated_numbers_flags_figures_absent_from_sources():
    sources = ["[NVDA] we repurchased 282 million shares for $40.4 billion"]
    assert fabricated_numbers("It repurchased 282 million shares [1]", sources) == set()
    assert fabricated_numbers("It repurchased 285 million shares [1]", sources) == {"285"}


def test_key_figure_present_ignores_parentheticals():
    expected = "$56,950 million (about $57.0 billion)."
    assert key_figure_present(expected, "equity was $56,950 million [1]") is True
    assert key_figure_present(expected, "equity was $57.0 billion [1]") is False  # paren-only
    assert key_figure_present("No figures here.", "whatever") is None


def test_parse_citations_range_check():
    assert parse_citations("A [1] and B [3].", 5) == ([1, 3], True)
    assert parse_citations("Out of range [7].", 5) == ([7], False)
    assert parse_citations("No citations.", 5) == ([], False)


def test_is_refusal_tolerates_wrapping():
    assert is_refusal("I cannot answer this from the provided filings.")
    assert is_refusal("Sorry — I cannot answer this from the provided filings, as none mention it.")
    assert not is_refusal("Revenue was $10 [1].")
