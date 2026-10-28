"""
Phase 2 - Chunking
-------------------
Turns the raw lines/tables from ingest.py into retrieval-ready chunks.

Chunking strategy: chunk by SECTION, not fixed token windows.
  - Headings are detected via font size (a line noticeably bigger than the
    document's median "body" font size, or bold-and-body-sized).
  - Everything between two headings becomes one prose chunk, tagged with
    its section title and page range.
  - Tables are kept as SEPARATE chunks (rendered to markdown) and tagged
    with whichever section they physically fall under, so a query like
    "UBRR bit layout" can hit the table directly instead of a paraphrase
    of it.

Real datasheets add three problems a clean PDF doesn't have, all handled here:
  - Repeated running headers/footers ("ATmega32(L)") get stripped before
    heading detection so they don't pollute titles.
  - Bare page numbers (a different string on every page, so frequency-based
    dedup can't catch them) get stripped by a "is this line just digits"
    check -- otherwise "7 Register Summary 8" is what you get when the
    page-7 footer number and page-8 header number merge into the title.
  - Runs of bold list items (e.g. numbered revision-history entries) can
    each look like a heading fragment; heading merging is capped so they
    can't snowball into one unreadable giant title.

This is a heuristic, not a PDF-structure parser -- expect to eyeball the
output and tighten `is_heading()` for your specific datasheet's layout.

Usage:
    python src/chunk_docs.py --doc-id atmega32
    python src/chunk_docs.py --doc-id atmega32 --exclude-sections "revision history,errata,ordering information,packaging information"
"""
import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

MAX_HEADING_CHARS = 120  # cap on merged heading length; see build_sections()
_PAGE_NUM_RE = re.compile(r"^\d{1,4}\.?$")
# Espressif-style register headers, e.g. "Register 1.1. SENS_ULP_CP_SLEEP_CYCn_REG (0x18+0x4*n)".
# This never occurs in body prose, so it's a safe additional heading signal
# regardless of font size -- added after this exact pattern was found NOT
# clearing the font-size threshold in the ESP32 TRM, collapsing 50+ distinct
# registers under one stale "Registers" chapter heading.
_REGISTER_HEADER_RE = re.compile(r"^Register\s+\d+\.\d+\.\s")


def _bbox_overlaps(b1, b2, tol: float = 2.0) -> bool:
    x0_1, y0_1, x1_1, y1_1 = b1
    x0_2, y0_2, x1_2, y1_2 = b2
    return not (x1_1 < x0_2 - tol or x0_1 > x1_2 + tol or
                y1_1 < y0_2 - tol or y0_1 > y1_2 + tol)


def strip_table_lines(pages: list[dict], tables: list[dict]) -> list[dict]:
    """Drop any text line that physically overlaps a detected table's bbox.

    Without this, table cell text gets extracted twice -- once as a table
    chunk, once as stray prose -- and the (usually small) table font sizes
    drag down the body-font-size estimate used for heading detection.
    """
    boxes_by_page: dict[int, list] = {}
    for t in tables:
        boxes_by_page.setdefault(t["page_num"], []).append(t["bbox"])

    cleaned = []
    for page in pages:
        boxes = boxes_by_page.get(page["page_num"], [])
        kept = [ln for ln in page["lines"]
                if not any(_bbox_overlaps(ln["bbox"], b) for b in boxes)]
        cleaned.append({"page_num": page["page_num"], "lines": kept})
    return cleaned


def strip_page_numbers(pages: list[dict]) -> tuple[list[dict], int]:
    """Drop lines that are just a bare page number (e.g. "7", "17", "12.").

    Unlike running headers, page numbers are a DIFFERENT string on every
    page, so frequency-based boilerplate detection can't catch them -- but
    they're often set in a bold/larger font (matching the running header or
    section-title style), which makes them look exactly like a heading
    fragment to is_heading(). Left unhandled, "7 Register Summary 8" is what
    you get when the page-7 footer number, the real title, and the page-8
    header number all get merged together.
    """
    cleaned = []
    removed = 0
    for page in pages:
        kept = []
        for ln in page["lines"]:
            if _PAGE_NUM_RE.match(ln["text"].strip()):
                removed += 1
                continue
            kept.append(ln)
        cleaned.append({"page_num": page["page_num"], "lines": kept})
    return cleaned, removed


def strip_repeated_boilerplate(pages: list[dict], min_page_fraction: float = 0.2) -> list[dict]:
    """Drop short lines that repeat verbatim across many pages.

    Running headers/footers (doc title, part number strings) are short and
    appear on most pages with identical text -- real section headings and
    body text don't repeat like that. Anything short enough AND common
    enough is almost certainly boilerplate, not content. (Page numbers are
    handled separately by strip_page_numbers, since each one is a distinct
    string and won't repeat.)
    """
    num_pages = len(pages) or 1
    counts = Counter()
    for page in pages:
        seen_this_page = {ln["text"] for ln in page["lines"] if len(ln["text"]) <= 40}
        counts.update(seen_this_page)

    threshold = max(3, int(num_pages * min_page_fraction))
    boilerplate = {text for text, n in counts.items() if n >= threshold}

    cleaned = []
    for page in pages:
        kept = [ln for ln in page["lines"] if ln["text"] not in boilerplate]
        cleaned.append({"page_num": page["page_num"], "lines": kept})
    return cleaned, boilerplate


def detect_body_font_size(pages: list[dict]) -> float:
    sizes = [ln["font_size"] for p in pages for ln in p["lines"]]
    return statistics.median(sizes) if sizes else 10.0


def is_heading(line: dict, body_size: float) -> bool:
    text = line["text"]
    # Document-specific structural marker, checked first and independent of
    # font size -- see _REGISTER_HEADER_RE comment above. Length-capped
    # defensively in case body prose ever coincidentally opens with this
    # exact "Register N.N." shape.
    if len(text) <= 150 and _REGISTER_HEADER_RE.match(text):
        return True
    bigger = line["font_size"] >= body_size + 1.0
    short = len(text) <= 90  # headings are short; wrapped sentences aren't
    has_letter = any(c.isalpha() for c in text)  # reject bare symbols/formula fragments
    return has_letter and short and (bigger or (line["bold"] and line["font_size"] >= body_size))


def build_sections(pages: list[dict], body_size: float) -> list[dict]:
    """Walk lines in reading order, grouping into (title, body) sections.

    Consecutive heading lines with no body text between them (e.g. a heading
    that word-wrapped across two lines) are merged into a single title --
    but capped at MAX_HEADING_CHARS so a run of bold list items (numbered
    revision-history entries, etc.) can't snowball into one giant title.
    Once the cap is hit, further "heading-looking" lines are treated as body
    text instead.
    """
    sections = []
    section_title = "Front Matter"
    body_parts: list[str] = []
    page_start, page_end = 1, 1

    def flush():
        if body_parts:
            sections.append({
                "title": section_title,
                "page_start": page_start,
                "page_end": page_end,
                "text": " ".join(body_parts),
            })

    for page in pages:
        page_num = page["page_num"]
        for line in page["lines"]:
            heading_like = is_heading(line, body_size)
            title_would_overflow = len(section_title) + 1 + len(line["text"]) > MAX_HEADING_CHARS

            if heading_like and not (title_would_overflow and not body_parts):
                if body_parts:
                    # body text since the last heading fragment -> real boundary
                    flush()
                    section_title = line["text"]
                    body_parts = []
                    page_start = page_num
                else:
                    # still the same (possibly multi-line) heading
                    if section_title == "Front Matter" and not sections:
                        section_title = line["text"]
                    else:
                        section_title += " " + line["text"]
                page_end = page_num
            else:
                # either genuinely body text, or a heading-like line that would
                # blow the merge cap -- treat as body content either way
                body_parts.append(line["text"])
                page_end = page_num

    flush()
    return sections


def table_to_markdown(rows: list[list]) -> str:
    rows = [[c if c is not None else "" for c in row] for row in rows]
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def nearest_section_title(sections: list[dict], page_num: int) -> str:
    best = "Unknown Section"
    for s in sections:
        if s["page_start"] <= page_num <= s["page_end"] + 1:
            best = s["title"]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--in-dir", default="data/processed")
    ap.add_argument("--min-chars", type=int, default=40,
                     help="Drop prose chunks shorter than this (bare headers, etc.)")
    ap.add_argument("--exclude-sections", default="",
                     help="Comma-separated, case-insensitive substrings; sections whose "
                          "title contains any of these are dropped. Useful for stripping "
                          "boilerplate not worth answering questions from, e.g. "
                          "'revision history,errata,ordering information,packaging information'")
    ap.add_argument("--exclude-tail", action="store_true",
                     help="Once a section matches --exclude-sections, also drop EVERY "
                          "section/table after it (by page), not just sections whose own "
                          "title matches. Administrative sections (errata, revision history, "
                          "ordering/packaging info) are almost always clustered at the end of "
                          "a datasheet and are made up of many small sub-items whose titles "
                          "don't repeat the parent heading -- keyword matching alone won't "
                          "catch 'Changes from Rev. 2503P-07/09' under a 'Revision History' "
                          "exclusion. This assumes nothing worth keeping comes after the "
                          "first match, so check where that match lands before relying on it.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    pages = json.loads((in_dir / f"{args.doc_id}_pages.json").read_text())
    tables = json.loads((in_dir / f"{args.doc_id}_tables.json").read_text())

    pages = strip_table_lines(pages, tables)
    pages, num_page_numbers = strip_page_numbers(pages)
    if num_page_numbers:
        print(f"[chunk] stripped {num_page_numbers} bare page-number line(s)")
    pages, boilerplate = strip_repeated_boilerplate(pages)
    if boilerplate:
        print(f"[chunk] stripped {len(boilerplate)} repeated header/footer line(s): "
              f"{sorted(boilerplate)[:5]}{'...' if len(boilerplate) > 5 else ''}")

    body_size = detect_body_font_size(pages)
    print(f"[chunk] detected body font size: {body_size}")

    sections = build_sections(pages, body_size)
    print(f"[chunk] detected {len(sections)} sections")

    exclude_terms = [t.strip().lower() for t in args.exclude_sections.split(",") if t.strip()]

    def excluded(title: str) -> bool:
        t = title.lower()
        return any(term in t for term in exclude_terms)

    cutoff_page = None
    if exclude_terms and args.exclude_tail:
        for s in sections:
            if excluded(s["title"]):
                cutoff_page = s["page_start"]
                print(f"[chunk] --exclude-tail: cutting everything from page "
                      f"{cutoff_page} onward (first match: {s['title']!r})")
                break

    def past_cutoff(page_num: int) -> bool:
        return cutoff_page is not None and page_num >= cutoff_page

    chunks = []
    cid = 0
    skipped_excluded = 0
    for s in sections:
        if len(s["text"]) < args.min_chars:
            continue
        if exclude_terms and (excluded(s["title"]) or past_cutoff(s["page_start"])):
            skipped_excluded += 1
            continue
        chunks.append({
            "chunk_id": f"{args.doc_id}_c{cid:04d}",
            "doc_id": args.doc_id,
            "chunk_type": "prose",
            "section_title": s["title"],
            "page_start": s["page_start"],
            "page_end": s["page_end"],
            "text": s["text"],
        })
        cid += 1

    for t in tables:
        title = nearest_section_title(sections, t["page_num"])
        if exclude_terms and (excluded(title) or past_cutoff(t["page_num"])):
            skipped_excluded += 1
            continue
        md = table_to_markdown(t["rows"])
        chunks.append({
            "chunk_id": f"{args.doc_id}_c{cid:04d}",
            "doc_id": args.doc_id,
            "chunk_type": "table",
            "section_title": title,
            "page_start": t["page_num"],
            "page_end": t["page_num"],
            "text": md,
        })
        cid += 1

    out_path = in_dir / f"{args.doc_id}_chunks.jsonl"
    with out_path.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    print(f"[chunk] wrote {len(chunks)} chunks -> {out_path}")
    print(f"[chunk]   prose chunks: {sum(1 for c in chunks if c['chunk_type']=='prose')}")
    print(f"[chunk]   table chunks: {sum(1 for c in chunks if c['chunk_type']=='table')}")
    if exclude_terms:
        print(f"[chunk]   excluded {skipped_excluded} chunk(s) matching {exclude_terms}")


if __name__ == "__main__":
    main()
