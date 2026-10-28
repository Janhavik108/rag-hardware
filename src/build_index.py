"""
Phase 3 - Build hybrid retrieval index
----------------------------------------
Builds two parallel indices over the same chunk set:
  1. Dense: sentence-transformers embeddings in a FAISS flat index (cosine
     similarity via normalized inner product).
  2. Sparse: BM25 over tokenized chunk text (rank_bm25).

Why hybrid, not just dense? Datasheets are full of exact register names and
numeric values (UBRR11, 0x1FF, ADPS2:0) that dense embeddings often blur
together semantically -- "UBRR11" and "UBRR10" can end up nearly identical
in embedding space even though they're different bits entirely. BM25 nails
exact-token matches but misses paraphrases/conceptual queries ("how do I
set the baud rate" vs the literal string "UBRR"). retrieve.py fuses both
rankings at query time so neither weakness dominates.

Both indices are built over the SAME chunk order, so row i in the FAISS
index and row i in the BM25 corpus always refer to the same chunk (see
_meta.json, which is that shared chunk list).

Usage:
    python src/build_index.py --doc-id atmega32
"""
import argparse
import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np

EMBED_MODEL = "all-MiniLM-L6-v2"  # small, fast, solid default for retrieval


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open() as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def embed_text(chunk: dict) -> str:
    """What actually gets embedded/tokenized for search.

    Prefixing with the section title gives the model context a bare chunk
    of register-table markdown wouldn't have on its own -- a table chunk's
    raw text might just be "| UBRRH | - | - | ... |" with no indication
    it's about baud rate registers unless the title comes along with it.
    """
    return f"{chunk['section_title']}: {chunk['text']}"


def tokenize(text: str) -> list[str]:
    """Simple alnum tokenizer. Keeps register names like 'UBRR11' as one
    token -- that's exactly the kind of exact match BM25 is here for."""
    return re.findall(r"[a-z0-9]+", text.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--in-dir", default="data/processed")
    ap.add_argument("--out-dir", default="data/index")
    ap.add_argument("--skip-dense", action="store_true",
                     help="Build only the BM25 index + metadata, skip the FAISS/embedding "
                          "step entirely. Useful for testing without model-download access, "
                          "or if you deliberately want a BM25-only baseline for comparison. "
                          "retrieve.py --bm25-only works fine against an index built this way; "
                          "retrieve.py in hybrid or --dense-only mode will not, since there's "
                          "no FAISS file yet.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(in_dir / f"{args.doc_id}_chunks.jsonl")
    print(f"[index] loaded {len(chunks)} chunks")

    # ---- Dense: FAISS ----
    if args.skip_dense:
        print("[index] --skip-dense set: not building the FAISS/embedding index")
    else:
        # Imported here, not at module level, so --skip-dense workflows (and the
        # mechanics test in scripts/test_retrieval_mechanics.py) don't need the
        # sentence-transformers package or a model download just to import this file.
        from sentence_transformers import SentenceTransformer

        print(f"[index] loading embedding model '{EMBED_MODEL}' (downloads on first run)...")
        model = SentenceTransformer(EMBED_MODEL)
        texts = [embed_text(c) for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)  # so inner product == cosine similarity

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, str(out_dir / f"{args.doc_id}.faiss"))
        print(f"[index] wrote FAISS index: {index.ntotal} vectors, dim={dim}")

    # ---- Sparse: BM25 ----
    tokenized_corpus = [tokenize(embed_text(c)) for c in chunks]
    with (out_dir / f"{args.doc_id}_bm25_corpus.pkl").open("wb") as f:
        pickle.dump(tokenized_corpus, f)
    print(f"[index] wrote BM25 tokenized corpus ({len(tokenized_corpus)} docs)")

    # ---- Shared metadata (row order matches both indices) ----
    with (out_dir / f"{args.doc_id}_meta.json").open("w") as f:
        json.dump(chunks, f)
    print(f"[index] wrote metadata for {len(chunks)} chunks")

    print(f"[index] done. Query with: "
          f"python src/retrieve.py --doc-id {args.doc_id} --query \"...\"")


if __name__ == "__main__":
    main()
