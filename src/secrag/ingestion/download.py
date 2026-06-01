"""Download 10-K primary documents from EDGAR into data/raw/."""

import argparse
import json
import logging
from pathlib import Path

from secrag.ingestion.edgar import EdgarClient, parse_10k_filings, primary_doc_url

log = logging.getLogger(__name__)


def download_10ks(
    tickers: list[str], years: int, dest: Path, client: EdgarClient | None = None
) -> list[Path]:
    client = client or EdgarClient()
    written: list[Path] = []
    for ticker in (t.upper() for t in tickers):
        cik = client.get_cik(ticker)
        submissions = client.get_submissions(cik)
        company_name = submissions.get("name", ticker)
        for filing in parse_10k_filings(submissions)[:years]:
            out = dest / ticker / f"FY{filing.fiscal_year}_{filing.accession_nodash}.html"
            if out.exists():
                log.info("skip (exists): %s", out)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(client.download_filing_html(cik, filing), encoding="utf-8")
            out.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "ticker": ticker,
                        "cik": cik,
                        "company_name": company_name,
                        "filing_type": "10-K",
                        "fiscal_year": filing.fiscal_year,
                        "accession_number": filing.accession_number,
                        "filing_date": filing.filing_date,
                        "report_date": filing.report_date,
                        "source_url": primary_doc_url(cik, filing),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            written.append(out)
            log.info("downloaded: %s", out)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Download 10-K filings from SEC EDGAR")
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--dest", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    written = download_10ks(args.tickers, args.years, args.dest)
    log.info("done: %d new file(s)", len(written))


if __name__ == "__main__":
    main()
