"""
Sanity-checks the retrieval MECHANICS (FAISS save/load, BM25 scoring, RRF
fusion math) using fake random embeddings instead of a real sentence-
transformers model. Useful for a quick check in an environment without
model-download access, or as a fast regression test that doesn't need GPU
time or a network call.

This does NOT test embedding quality/semantic search relevance -- only that
the plumbing (index build, save/load, search, fusion) works and doesn't
crash. Run build_index.py + retrieve.py for the real thing.

Usage:
    python scripts/test_retrieval_mechanics.py --doc-id test
"""
import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from build_index import embed_text, tokenize  # noqa: E402
from retrieve import bm25_search, reciprocal_rank_fusion  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402


def fake_embed(texts: list[str], dim: int = 384, seed: int = 0) -> np.ndarray:
    """Deterministic pseudo-embeddings: same text -> same vector, different
    text -> different vector. Stands in for a real model just to exercise
    the FAISS index build/search code path."""
    rng = np.random.default_rng(seed)
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        local_seed = abs(hash(t)) % (2**32)
        local_rng = np.random.default_rng(local_seed)
        vecs[i] = local_rng.normal(size=dim).astype(np.float32)
    return vecs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", default="test")
    ap.add_argument("--in-dir", default="data/processed")
    args = ap.parse_args()

    chunks_path = Path(args.in_dir) / f"{args.doc_id}_chunks.jsonl"
    if not chunks_path.exists():
        raise SystemExit(f"No chunks at {chunks_path} -- run chunk_docs.py first.")

    chunks = [json.loads(line) for line in chunks_path.open()]
    texts = [embed_text(c) for c in chunks]
    print(f"[test] loaded {len(chunks)} chunks")

    # ---- FAISS round-trip with fake embeddings ----
    embeddings = fake_embed(texts)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    tmp_path = Path("/tmp/test_retrieval.faiss")
    faiss.write_index(index, str(tmp_path))
    reloaded = faiss.read_index(str(tmp_path))
    assert reloaded.ntotal == len(chunks), "FAISS index save/load mismatch"
    print(f"[test] FAISS index round-trip OK ({reloaded.ntotal} vectors)")

    # Self-search sanity check: querying with a chunk's own embedding should
    # return that exact chunk as the #1 hit (cosine similarity ~1.0).
    query_vec = embeddings[0:1]
    scores, idxs = reloaded.search(query_vec, 1)
    assert idxs[0][0] == 0, "self-search did not return the query's own chunk as top hit"
    print(f"[test] FAISS self-search OK (score={scores[0][0]:.4f}, expect ~1.0)")

    # ---- BM25 ----
    tokenized_corpus = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    query = "UBRR baud rate register"
    results = bm25_search(bm25, query, k=len(chunks))
    print(f"\n[test] BM25 query {query!r}:")
    for idx, score in results:
        print(f"    [{chunks[idx]['chunk_id']}] score={score:.4f} "
              f"section={chunks[idx]['section_title']!r}")
    assert results, "BM25 returned zero results for a query containing corpus terms"
    print("[test] BM25 search OK (non-empty results for a term that's actually in the corpus)")

    # ---- RRF fusion ----
    fake_dense_results = [(i, float(1.0 - 0.1 * i)) for i in range(len(chunks))]
    fused = reciprocal_rank_fusion(fake_dense_results, results)
    assert len(fused) > 0, "RRF fusion produced no results"
    top_idx = max(fused, key=fused.get)
    print(f"\n[test] RRF fusion OK -- top fused result: [{chunks[top_idx]['chunk_id']}] "
          f"score={fused[top_idx]:.4f}")

    print("\n[test] ALL MECHANICS CHECKS PASSED")
    print("[test] (this does not validate embedding QUALITY -- run build_index.py + "
          "retrieve.py with the real model for that)")


if __name__ == "__main__":
    main()
