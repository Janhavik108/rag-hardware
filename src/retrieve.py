"""
Phase 3 - Hybrid retrieval
---------------------------
Fuses dense (FAISS) and sparse (BM25) rankings via Reciprocal Rank Fusion:

    score(doc) = sum over retrievers of 1 / (rrf_k + rank)

RRF is used instead of averaging raw scores because cosine similarity and
BM25 scores live on completely different, incomparable scales (cosine is
bounded [-1, 1]; BM25 is an unbounded, corpus-dependent score). RRF only
cares about each retriever's RANK ORDER, which sidesteps the normalization
problem entirely -- a standard, defensible choice for this kind of fusion,
and the reason it's what you'd cite if asked "why not just average the
scores" in an interview.

Usage:
    python src/retrieve.py --doc-id atmega32 --query "how is the baud rate calculated"
    python src/retrieve.py --doc-id atmega32 --query "UBRR11 bit" --top-k 5

    # for comparing hybrid against either retriever alone:
    python src/retrieve.py --doc-id atmega32 --query "..." --dense-only
    python src/retrieve.py --doc-id atmega32 --query "..." --bm25-only
"""
import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from build_index import EMBED_MODEL, tokenize


def load_faiss(doc_id: str, in_dir: Path):
    path = in_dir / f"{doc_id}.faiss"
    if not path.exists():
        raise SystemExit(
            f"No FAISS index at {path}. Build it with 'python src/build_index.py "
            f"--doc-id {doc_id}' (without --skip-dense), or use --bm25-only.")
    return faiss.read_index(str(path))


def load_bm25(doc_id: str, in_dir: Path) -> BM25Okapi:
    with (in_dir / f"{doc_id}_bm25_corpus.pkl").open("rb") as f:
        tokenized_corpus = pickle.load(f)
    return BM25Okapi(tokenized_corpus)


def load_meta(doc_id: str, in_dir: Path) -> list[dict]:
    with (in_dir / f"{doc_id}_meta.json").open() as f:
        return json.load(f)


def dense_search(model, index, query: str, k: int) -> list[tuple[int, float]]:
    qvec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(qvec)
    scores, idxs = index.search(qvec.astype(np.float32), k)
    return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]


def bm25_search(bm25: BM25Okapi, query: str, k: int) -> list[tuple[int, float]]:
    scores = bm25.get_scores(tokenize(query))
    top_idx = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]


def reciprocal_rank_fusion(dense: list[tuple[int, float]], bm25: list[tuple[int, float]],
                            rrf_k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
    for rank, (idx, _) in enumerate(bm25, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
    return fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--in-dir", default="data/index")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--fetch-k", type=int, default=20,
                     help="How many candidates each retriever pulls before fusion")
    ap.add_argument("--dense-only", action="store_true", help="Skip BM25, dense results only")
    ap.add_argument("--bm25-only", action="store_true", help="Skip dense, BM25 results only")
    args = ap.parse_args()
    if args.dense_only and args.bm25_only:
        raise SystemExit("--dense-only and --bm25-only are mutually exclusive")

    in_dir = Path(args.in_dir)
    meta = load_meta(args.doc_id, in_dir)
    bm25 = load_bm25(args.doc_id, in_dir)

    dense_results, bm25_results = [], []
    if not args.bm25_only:
        index = load_faiss(args.doc_id, in_dir)
        from sentence_transformers import SentenceTransformer
        print(f"[retrieve] loading embedding model '{EMBED_MODEL}'...")
        model = SentenceTransformer(EMBED_MODEL)
        dense_results = dense_search(model, index, args.query, args.fetch_k)
    if not args.dense_only:
        bm25_results = bm25_search(bm25, args.query, args.fetch_k)

    if args.dense_only:
        ranked = dense_results[:args.top_k]
    elif args.bm25_only:
        ranked = bm25_results[:args.top_k]
    else:
        fused_scores = reciprocal_rank_fusion(dense_results, bm25_results)
        ranked = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[:args.top_k]

    dense_rank = {idx: r for r, (idx, _) in enumerate(dense_results, start=1)}
    bm25_rank = {idx: r for r, (idx, _) in enumerate(bm25_results, start=1)}

    print(f"\nQuery: {args.query!r}  "
          f"(mode: {'dense-only' if args.dense_only else 'bm25-only' if args.bm25_only else 'hybrid'})\n")
    for rank, (idx, score) in enumerate(ranked, start=1):
        c = meta[idx]
        d_r = dense_rank.get(idx, "-")
        b_r = bm25_rank.get(idx, "-")
        print(f"#{rank} [{c['chunk_id']}] score={score:.4f}  (dense_rank={d_r}, bm25_rank={b_r})")
        print(f"    type={c['chunk_type']} section={c['section_title']!r} "
              f"pages={c['page_start']}-{c['page_end']}")
        print(f"    {c['text'][:200].strip()}")
        print()


if __name__ == "__main__":
    main()
