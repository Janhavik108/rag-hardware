"""
View/filter chunks produced by chunk_docs.py.

Examples:
    python scripts/view_chunks.py --doc-id atmega32
    python scripts/view_chunks.py --doc-id atmega32 --type table
    python scripts/view_chunks.py --doc-id atmega32 --section usart
    python scripts/view_chunks.py --doc-id atmega32 --id atmega32_c0012 --full
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--in-dir", default="data/processed")
    ap.add_argument("--type", choices=["prose", "table"], help="Filter by chunk_type")
    ap.add_argument("--section", help="Case-insensitive substring match on section_title")
    ap.add_argument("--id", help="Show only the chunk with this exact chunk_id")
    ap.add_argument("--full", action="store_true", help="Print full text instead of a 150-char preview")
    args = ap.parse_args()

    path = Path(args.in_dir) / f"{args.doc_id}_chunks.jsonl"
    if not path.exists():
        raise SystemExit(f"No chunks file found at {path} -- run chunk_docs.py first.")

    shown = 0
    with path.open() as f:
        for line in f:
            c = json.loads(line)

            if args.id and c["chunk_id"] != args.id:
                continue
            if args.type and c["chunk_type"] != args.type:
                continue
            if args.section and args.section.lower() not in c["section_title"].lower():
                continue

            shown += 1
            print(f"[{c['chunk_id']}] type={c['chunk_type']} "
                  f"section={c['section_title']!r} pages={c['page_start']}-{c['page_end']}")
            text = c["text"] if args.full else c["text"][:150].replace("\n", " ")
            print("  text:", text)
            print()

    print(f"-- {shown} chunk(s) shown --")


if __name__ == "__main__":
    main()
