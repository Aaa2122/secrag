"""SEC EDGAR access: fair-access compliant (declared UA, <=10 req/s, backoff)."""

import time
from dataclasses import dataclass

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from secrag.config import get_settings

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
MIN_INTERVAL_S = 0.11  # ~9 req/s, under SEC's 10 req/s fair-access limit


@dataclass(frozen=True)
class Filing:
    accession_number: str
    filing_date: str
    report_date: str
    primary_document: str
    form: str

    @property
    def fiscal_year(self) -> int:
        return int(self.report_date[:4])

    @property
    def accession_nodash(self) -> str:
        return self.accession_number.replace("-", "")


def parse_10k_filings(submissions: dict) -> list[Filing]:
    recent = submissions["filings"]["recent"]
    filings = [
        Filing(
            accession_number=recent["accessionNumber"][i],
            filing_date=recent["filingDate"][i],
            report_date=recent["reportDate"][i],
            primary_document=recent["primaryDocument"][i],
            form=recent["form"][i],
        )
        for i in range(len(recent["form"]))
        if recent["form"][i] == "10-K"
    ]
    return sorted(filings, key=lambda f: f.filing_date, reverse=True)


def primary_doc_url(cik: str, filing: Filing) -> str:
    return ARCHIVES_URL.format(
        cik_int=int(cik), acc_nodash=filing.accession_nodash, doc=filing.primary_document
    )


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and (
        exc.response.status_code == 429 or exc.response.status_code >= 500
    )


class EdgarClient:
    def __init__(self, user_agent: str | None = None) -> None:
        ua = user_agent or get_settings().sec_user_agent
        self._http = httpx.Client(headers={"User-Agent": ua}, timeout=30.0, follow_redirects=True)
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        wait = MIN_INTERVAL_S - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, max=30),
        reraise=True,
    )
    def _get(self, url: str) -> httpx.Response:
        self._throttle()
        resp = self._http.get(url)
        resp.raise_for_status()
        return resp

    def get_cik(self, ticker: str) -> str:
        data = self._get(TICKERS_URL).json()
        for entry in data.values():
            if entry["ticker"].upper() == ticker.upper():
                return f"{entry['cik_str']:010d}"
        raise KeyError(f"ticker not found on EDGAR: {ticker}")

    def get_submissions(self, cik: str) -> dict:
        return self._get(SUBMISSIONS_URL.format(cik=cik)).json()

    def download_filing_html(self, cik: str, filing: Filing) -> str:
        return self._get(primary_doc_url(cik, filing)).text
