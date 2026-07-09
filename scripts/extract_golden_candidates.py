"""Dump candidate chunks (UTF-8 file) from which golden quotes are hand-picked.

Usage: uv run python scripts/extract_golden_candidates.py OUT.md TICKER [TICKER...]
"""

import asyncio
import sys
from pathlib import Path

from sqlalchemy import text

from secrag.db import session_factory

PROBES = [
    (
        "numeric_sales",
        "7",
        ["%net sales%", "%total revenue%", "%revenue increased%", "%revenue decreased%"],
    ),
    (
        "numeric_other",
        None,
        ["%full-time employees%", "%dividend%per share%", "%repurchase%billion%"],
    ),
    ("risk_supply", "1a", ["%supply%", "%single source%", "%suppliers%"]),
    ("risk_ai", "1a", ["%artificial intelligence%", "% AI %"]),
    ("risk_reg", "1a", ["%regulat%", "%antitrust%", "%export control%"]),
    ("business", "1", ["%segment%", "%products%"]),
]

SQL = """
SELECT c.id, d.ticker, d.fiscal_year, c.metadata->>'item' AS item,
       (c.metadata->>'is_table')::bool AS is_table, left(c.content, 650) AS excerpt
FROM chunks c JOIN documents d ON c.document_id = d.id
WHERE d.ticker = :ticker
  AND (CAST(:item AS text) IS NULL OR c.metadata->>'item' = CAST(:item AS text))
  AND ({likes})
ORDER BY c.token_count DESC
LIMIT :n
"""


async def main(out: Path, tickers: list[str]) -> None:
    lines: list[str] = []
    async with session_factory()() as session:
        for ticker in tickers:
            lines.append(f"\n{'=' * 30} {ticker} {'=' * 30}")
            for name, item, likes in PROBES:
                clause = " OR ".join(f"c.content ILIKE :like{i}" for i in range(len(likes)))
                params = {f"like{i}": pat for i, pat in enumerate(likes)}
                rows = (
                    await session.execute(
                        text(SQL.format(likes=clause)),
                        {"ticker": ticker, "item": item, "n": 2, **params},
                    )
                ).all()
                for r in rows:
                    lines.append(
                        f"\n--- {name} | chunk {r.id} | {r.ticker} FY{r.fiscal_year} "
                        f"| item={r.item} | table={r.is_table}\n{r.excerpt}"
                    )
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(lines)} blocks)")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), [t.upper() for t in sys.argv[2:]]))
