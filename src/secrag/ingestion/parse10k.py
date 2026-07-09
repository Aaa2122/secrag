"""Item-aware parsing of SEC 10-K primary documents (HTML).

Heading detection rules, established by probing real filings (see plan doc):
a real Item heading is a short (<120 chars) standalone block whose text starts
with "Item N", outside any <table>; the table-of-contents lives in a <table>
(excluded), and cross-references sit inside long paragraphs (excluded by
length). Item ids must appear in ascending 10-K order — out-of-order matches
are treated as body text, not headings.
"""

import re
from dataclasses import dataclass, field

from selectolax.parser import HTMLParser, Node

Block = tuple[str, str]  # (kind: "text" | "table", content)

ITEM_ORDER = [
    "1", "1a", "1b", "1c", "2", "3", "4", "5", "6", "7", "7a",
    "8", "9", "9a", "9b", "9c", "10", "11", "12", "13", "14", "15", "16",
]  # fmt: skip

CANONICAL_TITLES = {
    "1": "Business",
    "1a": "Risk Factors",
    "1b": "Unresolved Staff Comments",
    "1c": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Reserved",
    "7": "Management's Discussion and Analysis",
    "7a": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9a": "Controls and Procedures",
    "9b": "Other Information",
    "9c": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

ITEM_HEADING_RE = re.compile(r"^item\s+(\d{1,2}[a-c]?)\s*[.:–—-]?\s*(.*)$", re.IGNORECASE)
MAX_HEADING_LEN = 120

_WS_RE = re.compile(r"[ \t ]+")


@dataclass
class Section:
    item: str | None  # normalized lowercase id ("1a") or None for the preamble
    title: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(b[1]) for b in self.blocks)


def _norm_text(text: str) -> str:
    return _WS_RE.sub(" ", text.replace("​", "")).strip()


def _is_hidden(node: Node) -> bool:
    style = (node.attributes.get("style") or "").replace(" ", "").lower()
    return "display:none" in style


BLOCK_TAGS = {"p", "div", "table", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol"}
SKIP_TAGS = {"script", "style", "head", "title"}


def _walk_blocks(node: Node, out: list[Block]) -> None:
    """Emit leaf-most block elements in document order; tables are atomic."""
    for child in node.iter(include_text=False):
        tag = child.tag
        if tag in SKIP_TAGS or _is_hidden(child):
            continue
        if tag == "table":
            out.extend(_table_blocks(child))
            continue
        has_block_children = any(
            g.tag in BLOCK_TAGS for g in child.iter(include_text=False)
        ) or child.css_first("table") is not None
        if has_block_children:
            _walk_blocks(child, out)
        else:
            text = _norm_text(child.text(separator=" "))
            if text:
                out.append(("text", text))


def _table_blocks(node: Node) -> list[Block]:
    """Serialize a <table> to markdown; layout tables degrade to text blocks."""
    rows: list[list[str]] = []
    for tr in node.css("tr"):
        cells = [_norm_text(td.text(separator=" ")) for td in tr.css("td, th")]
        rows.append(cells)
    if not rows:
        return []

    width = max(len(r) for r in rows)
    for r in rows:
        r.extend([""] * (width - len(r)))
    keep = [i for i in range(width) if any(r[i] for r in rows)]
    rows = [[r[i] for i in keep] for r in rows if any(r[i] for i in keep)]
    if not rows:
        return []

    # Layout table (single effective column): emit rows as plain text blocks so
    # that Item headings wrapped in layout tables are still detectable.
    if len(keep) <= 1 or len(rows) < 2:
        return [("text", " ".join(c for c in row if c)) for row in rows if any(row)]

    header, body = rows[0], rows[1:]
    md = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
        *("| " + " | ".join(row) + " |" for row in body),
    ]
    return [("table", "\n".join(md))]


def _match_item_heading(text: str) -> tuple[str, str] | None:
    if len(text) > MAX_HEADING_LEN:
        return None
    m = ITEM_HEADING_RE.match(text)
    if not m:
        return None
    item = m.group(1).lower()
    if item not in ITEM_ORDER:
        return None
    title = m.group(2).strip(" .:–—-") or CANONICAL_TITLES.get(item, "")
    return item, title


def parse_10k(html: str) -> list[Section]:
    tree = HTMLParser(html)
    body = tree.body if tree.body is not None else tree.root
    blocks: list[Block] = []
    _walk_blocks(body, blocks)

    sections = [Section(item=None, title="Cover")]
    last_idx = -1
    for kind, content in blocks:
        heading = _match_item_heading(content) if kind == "text" else None
        if heading is not None:
            idx = ITEM_ORDER.index(heading[0])
            if idx > last_idx:
                sections.append(Section(item=heading[0], title=heading[1]))
                last_idx = idx
                continue
        sections[-1].blocks.append((kind, content))
    return sections


def parse_report(sections: list[Section]) -> dict:
    items = [s.item for s in sections if s.item is not None]
    core = {"1", "1a", "7", "8"}
    return {
        "items_found": items,
        "chars_per_item": {s.item or "cover": s.char_count for s in sections},
        "n_tables": sum(1 for s in sections for k, _ in s.blocks if k == "table"),
        "missing_core_items": sorted(core - set(items)),
    }
