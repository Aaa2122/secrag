import json

import httpx
import respx

from secrag.ingestion.download import download_10ks
from secrag.ingestion.edgar import EdgarClient

TICKERS_JSON = {"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"}}
SUBMISSIONS = {
    "name": "NVIDIA CORP",
    "filings": {
        "recent": {
            "accessionNumber": ["0001045810-25-000023", "0001045810-24-000030"],
            "form": ["10-K", "10-K"],
            "filingDate": ["2025-02-26", "2024-02-21"],
            "reportDate": ["2025-01-26", "2024-01-28"],
            "primaryDocument": ["nvda-20250126.htm", "nvda-20240128.htm"],
        }
    },
}


@respx.mock
def test_download_10ks_writes_files_and_sidecars(tmp_path):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=TICKERS_JSON)
    )
    respx.get("https://data.sec.gov/submissions/CIK0001045810.json").mock(
        return_value=httpx.Response(200, json=SUBMISSIONS)
    )
    respx.get(url__startswith="https://www.sec.gov/Archives/").mock(
        return_value=httpx.Response(200, text="<html>fake 10-K</html>")
    )

    client = EdgarClient(user_agent="test test@example.com")
    written = download_10ks(["NVDA"], years=1, dest=tmp_path, client=client)

    assert len(written) == 1
    html_path = tmp_path / "NVDA" / "FY2025_000104581025000023.html"
    assert html_path.read_text(encoding="utf-8") == "<html>fake 10-K</html>"
    sidecar = json.loads(html_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["ticker"] == "NVDA"
    assert sidecar["cik"] == "0001045810"
    assert sidecar["fiscal_year"] == 2025

    # idempotent: second run downloads nothing new
    assert download_10ks(["NVDA"], years=1, dest=tmp_path, client=client) == []
