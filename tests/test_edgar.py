import httpx
import pytest
import respx

from secrag.ingestion.edgar import EdgarClient, Filing, parse_10k_filings, primary_doc_url

SUBMISSIONS = {
    "cik": 1045810,
    "name": "NVIDIA CORP",
    "filings": {
        "recent": {
            "accessionNumber": [
                "0001045810-25-000023",
                "0001045810-24-000029",
                "0001045810-24-000030",
            ],
            "form": ["10-K", "10-K/A", "10-K"],
            "filingDate": ["2025-02-26", "2024-05-01", "2024-02-21"],
            "reportDate": ["2025-01-26", "2024-01-28", "2024-01-28"],
            "primaryDocument": ["nvda-20250126.htm", "nvda-20240128a.htm", "nvda-20240128.htm"],
        }
    },
}


def test_parse_10k_filings_filters_and_orders():
    filings = parse_10k_filings(SUBMISSIONS)
    assert [f.form for f in filings] == ["10-K", "10-K"]
    assert filings[0].accession_number == "0001045810-25-000023"
    assert filings[0].fiscal_year == 2025
    assert filings[1].fiscal_year == 2024


def test_primary_doc_url():
    filing = Filing(
        accession_number="0001045810-25-000023",
        filing_date="2025-02-26",
        report_date="2025-01-26",
        primary_document="nvda-20250126.htm",
        form="10-K",
    )
    assert primary_doc_url("0001045810", filing) == (
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/nvda-20250126.htm"
    )


@respx.mock
def test_get_cik_pads_to_ten_digits():
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(
            200,
            json={"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"}},
        )
    )
    client = EdgarClient(user_agent="test-agent test@example.com")
    assert client.get_cik("nvda") == "0001045810"
    with pytest.raises(KeyError):
        client.get_cik("NOPE")


@respx.mock
def test_download_retries_on_429_then_succeeds():
    route = respx.get(
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/nvda-20250126.htm"
    ).mock(side_effect=[httpx.Response(429), httpx.Response(200, text="<html>10-K</html>")])
    client = EdgarClient(user_agent="test-agent test@example.com")
    filing = Filing(
        accession_number="0001045810-25-000023",
        filing_date="2025-02-26",
        report_date="2025-01-26",
        primary_document="nvda-20250126.htm",
        form="10-K",
    )
    html = client.download_filing_html("0001045810", filing)
    assert "10-K" in html
    assert route.call_count == 2
