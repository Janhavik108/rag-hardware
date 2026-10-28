"""
Phase 6 - Evaluation harness
-------------------------------
Runs a hand-labeled eval set (data/eval/<doc_id>_eval.jsonl) through:
  1. Retrieval alone -- precision/recall against hand-labeled relevance.
  2. The full agent + verifier -- answer faithfulness (what fraction of the
     drafted answer's claims are actually supported by retrieved evidence)
     and tool-use compliance (for questions that require a calculator tool,
     did the agent actually call it, or compute the value itself?).

This is the piece the project brief calls the single biggest
differentiator -- almost nobody builds an eval harness for a student RAG
project, and these numbers are the actual answer to "how do you know it's
not hallucinating" instead of a shrug.

Ground truth uses KEYWORD matching, not exact chunk_ids. Chunk IDs are
reassigned every time chunk_docs.py runs (different --exclude flags, a
re-ingested PDF, a chunking heuristic tweak all shift them), so hardcoding
"the answer should be in atmega32_c0362" breaks the eval set every time the
pipeline changes upstream. "The answer should be in a chunk whose title or
text mentions 'UBRR' and 'baud rate'" doesn't. It's also just easier to
hand-write correctly while reading a datasheet -- you think in terms of
what a section is ABOUT, not what number it happens to have been assigned.

Usage:
    python src/eval.py --doc-id atmega32 --eval-set data/eval/atmega32_eval.jsonl
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import get_llm_client, run_agent  # noqa: E402
from verify import verify_answer  # noqa: E402

# Rough, imperfect heuristic for "did the answer refuse instead of fabricating
# a claim" on questions the corpus genuinely can't answer. Not a substitute
# for reading the answer yourself -- see summarize()'s printed reminder.
_REFUSAL_PHRASES = [
    "cannot determine", "can't determine", "no information", "not described",
    "doesn't contain", "does not contain", "not mentioned", "not covered",
    "cannot find", "can't find", "not found in", "no mention",
]


def chunk_matches_keywords(chunk: dict, keywords: list[str]) -> bool:
    """A chunk counts as relevant if ALL given keywords appear (case-
    insensitive substring match) in its section title + text combined."""
    haystack = (chunk.get("section_title", "") + " " + chunk.get("text", "")).lower()
    return all(kw.lower() in haystack for kw in keywords)


def retrieval_metrics(retrieved_chunks: list[dict], all_chunks: list[dict],
                       keywords: list[str]) -> dict:
    """Precision: of what was retrieved, how much is relevant?
    Recall: of everything relevant in the corpus, how much got retrieved?
    Both None if the question has no keywords (e.g. a deliberate
    refusal-test question with nothing relevant to retrieve)."""
    if not keywords:
        return {"precision": None, "recall": None, "total_relevant_in_corpus": None}

    retrieved_relevant = sum(1 for c in retrieved_chunks if chunk_matches_keywords(c, keywords))
    total_relevant = sum(1 for c in all_chunks if chunk_matches_keywords(c, keywords))

    precision = (retrieved_relevant / len(retrieved_chunks)) if retrieved_chunks else 0.0
    recall = (retrieved_relevant / total_relevant) if total_relevant else None

    return {"precision": precision, "recall": recall, "total_relevant_in_corpus": total_relevant}


def looks_like_refusal(answer: str) -> bool:
    a = (answer or "").lower()
    return any(p in a for p in _REFUSAL_PHRASES)


def run_eval(doc_id: str, eval_path: Path, index_dir: Path, top_k: int = 5,
             client=None, model_name: str = None, embed_model=None,
             verbose: bool = True, delay: float = 15.0, skip_ids: set = None,
             on_result=None) -> list[dict]:
    """
    `skip_ids`: question ids to skip entirely (already completed in a prior,
    interrupted run -- see --resume). `on_result`: called with each result
    dict immediately after that question finishes, so a caller can persist
    progress to disk incrementally rather than losing everything already
    computed if a later question hits an unrecoverable error (e.g. a daily
    token quota, which no amount of in-process retrying can wait out).
    """
    eval_items = [json.loads(line) for line in Path(eval_path).open()]
    if skip_ids:
        eval_items = [item for item in eval_items if item["id"] not in skip_ids]
        if verbose and eval_items:
            print(f"[eval] resuming: {len(skip_ids)} question(s) already done, "
                  f"{len(eval_items)} remaining")

    chunks_path = index_dir.parent / "processed" / f"{doc_id}_chunks.jsonl"
    all_chunks = [json.loads(line) for line in chunks_path.open()] if chunks_path.exists() else []
    if not all_chunks and verbose:
        print(f"[eval] warning: couldn't find {chunks_path} -- recall will be unavailable "
              f"(precision still works)")

    client = client or get_llm_client()

    results = []
    for i, item in enumerate(eval_items):
        if i > 0 and delay > 0:
            time.sleep(delay)

        qid = item["id"]
        question = item["question"]
        keywords = item.get("relevant_keywords", [])
        requires_tool = item.get("requires_tool", False)
        expected_tool = item.get("expected_tool")
        expect_refusal = item.get("expect_refusal", False)

        if verbose:
            print(f"[eval] {qid}: {question}")

        agent_result = run_agent(doc_id, question, top_k, index_dir, client=client,
                                  model_name=model_name, embed_model=embed_model, verbose=False)
        rmetrics = retrieval_metrics(agent_result["retrieved_chunks"], all_chunks, keywords)

        verification = verify_answer(question, agent_result["answer"],
                                      agent_result["retrieved_chunks"], agent_result["tool_calls"],
                                      client=client, model_name=model_name)
        claims = verification.get("claims", [])
        n_supported = sum(1 for c in claims if c.get("verdict") == "supported")
        faithfulness = (n_supported / len(claims)) if claims else None

        tool_names_called = {tc["name"] for tc in agent_result["tool_calls"]}
        tool_used_correctly = None
        if requires_tool:
            tool_used_correctly = (expected_tool in tool_names_called) if expected_tool \
                else bool(tool_names_called)

        refusal_check = looks_like_refusal(agent_result["answer"]) if expect_refusal else None

        result = {
            "id": qid, "question": question,
            "retrieval_precision": rmetrics["precision"], "retrieval_recall": rmetrics["recall"],
            "total_relevant_in_corpus": rmetrics["total_relevant_in_corpus"],
            "verification_overall": verification["overall"],
            "faithfulness": faithfulness, "n_claims": len(claims), "n_supported": n_supported,
            "claims": claims,
            "requires_tool": requires_tool, "tool_used_correctly": tool_used_correctly,
            "tool_calls_made": sorted(tool_names_called),
            "expect_refusal": expect_refusal, "looked_like_refusal": refusal_check,
            "answer": agent_result["answer"],
        }
        results.append(result)
        if on_result:
            on_result(result)

        if verbose:
            p = f"{rmetrics['precision']:.2f}" if rmetrics["precision"] is not None else "-"
            r = f"{rmetrics['recall']:.2f}" if rmetrics["recall"] is not None else "-"
            f_ = f"{faithfulness:.2f}" if faithfulness is not None else "-"
            print(f"    precision={p} recall={r} verification={verification['overall']} "
                  f"faithfulness={f_}")

    return results


def summarize(results: list[dict]) -> dict:
    precisions = [r["retrieval_precision"] for r in results if r["retrieval_precision"] is not None]
    recalls = [r["retrieval_recall"] for r in results if r["retrieval_recall"] is not None]
    faithfulnesses = [r["faithfulness"] for r in results if r["faithfulness"] is not None]

    n = len(results)
    passes = sum(1 for r in results if r["verification_overall"] == "pass")
    flagged = sum(1 for r in results if r["verification_overall"] in ("flagged", "flagged_partial"))
    verification_failed = sum(1 for r in results if r["verification_overall"] == "verification_failed")

    tool_required = [r for r in results if r["requires_tool"]]
    tool_compliance = (
        sum(1 for r in tool_required if r["tool_used_correctly"]) / len(tool_required)
    ) if tool_required else None

    refusal_items = [r for r in results if r["expect_refusal"]]
    refusal_rate = (
        sum(1 for r in refusal_items if r["looked_like_refusal"]) / len(refusal_items)
    ) if refusal_items else None

    return {
        "n_questions": n,
        "mean_retrieval_precision": round(statistics.mean(precisions), 3) if precisions else None,
        "mean_retrieval_recall": round(statistics.mean(recalls), 3) if recalls else None,
        "mean_answer_faithfulness": round(statistics.mean(faithfulnesses), 3) if faithfulnesses else None,
        "verification_pass_rate": round(passes / n, 3) if n else None,
        "verification_flagged_rate": round(flagged / n, 3) if n else None,
        "verification_failed_count": verification_failed,
        "tool_use_compliance_rate": round(tool_compliance, 3) if tool_compliance is not None else None,
        "n_tool_required_questions": len(tool_required),
        "refusal_heuristic_rate": round(refusal_rate, 3) if refusal_rate is not None else None,
        "n_refusal_test_questions": len(refusal_items),
    }


def _load_existing_results(out_path: Path) -> dict:
    """Returns {question_id: result_dict} from a previous run's output file,
    or {} if there isn't one / it's unreadable. Keyed by id so results can
    be merged and re-ordered regardless of what order they were computed in."""
    if not out_path.exists():
        return {}
    try:
        data = json.loads(out_path.read_text())
        return {r["id"]: r for r in data.get("results", [])}
    except (json.JSONDecodeError, KeyError, OSError):
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="data/eval/results.json")
    ap.add_argument("--delay", type=float, default=15.0,
                     help="Seconds to wait between questions, to avoid hitting free-tier "
                          "rate limits as often (each question makes 2-3 LLM calls). "
                          "Bumped up from an earlier 3s default that wasn't enough headroom "
                          "under an observed 8000 TPM cap and led to full retry exhaustion "
                          "on a 7-question run. Set to 0 to disable.")
    ap.add_argument("--resume", action="store_true",
                     help="Skip questions already present in --out from a previous run and "
                          "pick up where it left off, instead of starting over. Add this after "
                          "any interrupted run -- in particular, a provider's DAILY token quota "
                          "(distinct from the per-minute one --delay/retries handle) can't be "
                          "waited out with more retries, only resumed later once it refills.")
    args = ap.parse_args()

    out_path = Path(args.out)
    existing = _load_existing_results(out_path) if args.resume else {}
    if existing:
        print(f"[eval] --resume: found {len(existing)} previously-completed question(s) in {out_path}")

    embed_model = None
    try:
        from sentence_transformers import SentenceTransformer
        print("[eval] loading embedding model once for the whole run...")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        pass  # run_agent will load it per-question as a fallback if this isn't available

    eval_item_order = [json.loads(line)["id"] for line in Path(args.eval_set).open()]

    def save_progress():
        """Called after every question -- if the process dies on question 15
        of 22 (rate limit, network blip, Ctrl+C), questions 1-14 are already
        safely on disk and --resume picks up at 15, not 1."""
        ordered = [existing[qid] for qid in eval_item_order if qid in existing]
        summary = summarize(ordered)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": summary, "results": ordered}, indent=2))

    def on_result(result):
        existing[result["id"]] = result
        save_progress()

    run_eval(args.doc_id, Path(args.eval_set), Path(args.index_dir), args.top_k,
              embed_model=embed_model, delay=args.delay,
              skip_ids=set(existing.keys()), on_result=on_result)

    ordered_results = [existing[qid] for qid in eval_item_order if qid in existing]
    summary = summarize(ordered_results)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nFull per-question results written to {out_path}")
    if len(ordered_results) < len(eval_item_order):
        missing = len(eval_item_order) - len(ordered_results)
        print(f"\n{missing} question(s) not yet completed -- re-run with --resume to continue "
              f"once whatever blocked it (rate limit, etc.) has cleared.")
    if summary.get("n_refusal_test_questions"):
        print("\nNote: refusal_heuristic_rate is a rough keyword check, not a real judgment --\n"
              "read the actual answers for refusal-test questions in the results file yourself.")


if __name__ == "__main__":
    main()
