"""
Phase 2 - Ingestion
--------------------
Extracts, per page:
  1. Text spans WITH font size + bold flag (needed later for heading detection)
  2. Tables (register maps, pin tables, etc.) via ruled-line detection

Why two libraries?
  - PyMuPDF (fitz) exposes per-span font metadata, which is the cheapest
    reliable signal for "is this a heading" in a datasheet.
  - pdfplumber's table detection (based on ruling lines) is more accurate
    for the boxed register tables datasheets are full of than trying to
    infer table structure from raw text.

Usage:
    python src/ingest.py --pdf data/raw/atmega32.pdf --doc-id atmega32
"""
import argparse
import json
from pathlib import Path

import pymupdf  # PyMuPDF (the `fitz` import name is now deprecated)
import pdfplumber


def extract_lines(pdf_path: str) -> list[dict]:
    """Return [{page_num, lines: [{text, font_size, bold, bbox}]}] for the whole doc.

    Spans are merged into their PARENT LINE rather than kept individually.
    This matters for two reasons:
      1. A heading that's really one visual line (the common case) no longer
         gets split into fragments that have to be re-merged later.
      2. `bold` reflects whether the WHOLE line is bold, not just one word --
         a single emphasized word inside a normal paragraph line no longer
         gets mistaken for a heading.
    """
    doc = pymupdf.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        lines_out = []
        raw = page.get_text("dict")
        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                spans = [s for s in line.get("spans", []) if s["text"].strip()]
                if not spans:
                    continue
                text = " ".join(s["text"].strip() for s in spans)
                text = " ".join(text.split())  # collapse repeated whitespace
                bbox = (
                    min(s["bbox"][0] for s in spans),
                    min(s["bbox"][1] for s in spans),
                    max(s["bbox"][2] for s in spans),
                    max(s["bbox"][3] for s in spans),
                )
                lines_out.append({
                    "text": text,
                    "font_size": round(max(s["size"] for s in spans), 1),
                    "bold": all(bool(s["flags"] & 2 ** 4) for s in spans),
                    "bbox": bbox,
                })
        pages.append({"page_num": page_num, "lines": lines_out})
    doc.close()
    return pages


def _is_junk_table(rows: list[list]) -> bool:
    """Filter out pdfplumber detections that are really diagram borders, not tables.

    Datasheets are full of boxed pinout/block diagrams; pdfplumber's ruled-line
    table finder frequently mistakes the box outline for a table grid, producing
    a "table" whose cells are empty or near-empty. Real register/pin tables have
    substantial text in most cells; decorative boxes don't.
    """
    total_chars = sum(len((cell or "").strip()) for row in rows for cell in row)
    total_cells = sum(len(row) for row in rows)
    non_empty_cells = sum(1 for row in rows for cell in row if (cell or "").strip())
    if total_chars < 15:
        return True
    if total_cells and (non_empty_cells / total_cells) < 0.3:
        return True
    return False


def extract_tables(pdf_path: str) -> list[dict]:
    """Return [{page_num, table_index, bbox, rows}] using pdfplumber's ruled-table finder."""
    tables_out = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for t_idx, table in enumerate(page.find_tables()):
                data = table.extract()
                if not data or len(data) < 2:  # skip junk/near-empty detections
                    continue
                if _is_junk_table(data):
                    continue
                tables_out.append({
                    "page_num": page_num,
                    "table_index": t_idx,
                    "bbox": table.bbox,
                    "rows": data,
                })
    return tables_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to source datasheet PDF")
    ap.add_argument("--doc-id", required=True, help="Short id, e.g. atmega32")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ingest] extracting text lines from {args.pdf} ...")
    pages = extract_lines(args.pdf)
    pages_path = out_dir / f"{args.doc_id}_pages.json"
    pages_path.write_text(json.dumps(pages, indent=2))
    print(f"[ingest] wrote {pages_path}  ({len(pages)} pages)")

    print(f"[ingest] extracting tables from {args.pdf} ...")
    tables = extract_tables(args.pdf)
    tables_path = out_dir / f"{args.doc_id}_tables.json"
    tables_path.write_text(json.dumps(tables, indent=2))
    print(f"[ingest] wrote {tables_path}  ({len(tables)} tables found)")


if __name__ == "__main__":
    main()
