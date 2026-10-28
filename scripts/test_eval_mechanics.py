"""
Sanity-checks eval.py: keyword-based relevance matching, precision/recall
computation against a small fake corpus, refusal heuristic, and summary
aggregation -- using mocked run_agent/verify_answer, no API key or network
needed.

Usage:
    python scripts/test_eval_mechanics.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import eval as ev  # noqa: E402  (avoid shadowing the builtin `eval`)

failures = []


def check(label, actual, expected):
    ok = actual == expected
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: got {actual!r}, expected {expected!r}")
    if not ok:
        failures.append(label)


print("=== chunk_matches_keywords ===")

c1 = {"section_title": "USART Baud Rate Register", "text": "UBRR sets the baud rate."}
c2 = {"section_title": "Port A", "text": "Port A serves as analog input."}

check("matches when all keywords present", ev.chunk_matches_keywords(c1, ["UBRR", "baud"]), True)
check("case-insensitive match", ev.chunk_matches_keywords(c1, ["ubrr", "BAUD"]), True)
check("fails when one keyword missing", ev.chunk_matches_keywords(c1, ["UBRR", "prescaler"]), False)
check("unrelated chunk doesn't match", ev.chunk_matches_keywords(c2, ["UBRR", "baud"]), False)

print("\n=== retrieval_metrics ===")

all_chunks = [
    {"chunk_id": "c1", "section_title": "USART Baud Rate Register", "text": "UBRR sets the baud rate."},
    {"chunk_id": "c2", "section_title": "Baud Rate Examples", "text": "Example UBRR and baud values."},
    {"chunk_id": "c3", "section_title": "Port A", "text": "Port A serves as analog input."},
    {"chunk_id": "c4", "section_title": "Port B", "text": "Port B is bidirectional."},
]
# 2 of 4 corpus chunks are relevant to "UBRR"+"baud" (c1, c2)

retrieved_all_relevant = [all_chunks[0], all_chunks[1]]  # both relevant, none irrelevant
m = ev.retrieval_metrics(retrieved_all_relevant, all_chunks, ["UBRR", "baud"])
check("precision=1.0 when all retrieved are relevant", m["precision"], 1.0)
check("recall=1.0 when all relevant chunks retrieved", m["recall"], 1.0)
check("total_relevant_in_corpus counted correctly", m["total_relevant_in_corpus"], 2)

retrieved_mixed = [all_chunks[0], all_chunks[2]]  # 1 relevant (c1), 1 not (c3)
m = ev.retrieval_metrics(retrieved_mixed, all_chunks, ["UBRR", "baud"])
check("precision=0.5 with 1 relevant of 2 retrieved", m["precision"], 0.5)
check("recall=0.5 with 1 of 2 relevant chunks retrieved", m["recall"], 0.5)

retrieved_none_relevant = [all_chunks[2], all_chunks[3]]
m = ev.retrieval_metrics(retrieved_none_relevant, all_chunks, ["UBRR", "baud"])
check("precision=0.0 when nothing retrieved is relevant", m["precision"], 0.0)
check("recall=0.0 when nothing relevant was retrieved", m["recall"], 0.0)

m = ev.retrieval_metrics([], all_chunks, [])
check("empty keywords -> precision is None (no ground truth to score against)", m["precision"], None)
check("empty keywords -> recall is None", m["recall"], None)

print("\n=== looks_like_refusal ===")
check("detects a real refusal", ev.looks_like_refusal("I cannot determine this from the excerpts."), True)
check("doesn't flag a normal confident answer", ev.looks_like_refusal("UBRR = 51."), False)
check("handles None/empty gracefully", ev.looks_like_refusal(None), False)

print("\n=== summarize ===")

fake_results = [
    {"retrieval_precision": 1.0, "retrieval_recall": 1.0, "faithfulness": 1.0,
     "verification_overall": "pass", "requires_tool": True, "tool_used_correctly": True,
     "expect_refusal": False, "looked_like_refusal": None},
    {"retrieval_precision": 0.5, "retrieval_recall": 0.5, "faithfulness": 0.5,
     "verification_overall": "flagged", "requires_tool": True, "tool_used_correctly": False,
     "expect_refusal": False, "looked_like_refusal": None},
    {"retrieval_precision": None, "retrieval_recall": None, "faithfulness": None,
     "verification_overall": "pass", "requires_tool": False, "tool_used_correctly": None,
     "expect_refusal": True, "looked_like_refusal": True},
]
s = ev.summarize(fake_results)
check("n_questions counted", s["n_questions"], 3)
check("mean_retrieval_precision averages only non-None entries", s["mean_retrieval_precision"], 0.75)
check("verification_pass_rate = 2/3", s["verification_pass_rate"], round(2 / 3, 3))
check("verification_flagged_rate = 1/3", s["verification_flagged_rate"], round(1 / 3, 3))
check("tool_use_compliance_rate = 1/2 (one of two tool-required items succeeded)",
      s["tool_use_compliance_rate"], 0.5)
check("n_tool_required_questions counted", s["n_tool_required_questions"], 2)
check("refusal_heuristic_rate = 1/1", s["refusal_heuristic_rate"], 1.0)
check("n_refusal_test_questions counted", s["n_refusal_test_questions"], 1)

print("\n=== run_eval end-to-end with mocked agent/verify ===")

import agent as agent_module  # noqa: E402
import verify as verify_module  # noqa: E402

FAKE_CHUNKS = [{"chunk_id": "c1", "section_title": "USART Baud Rate Register",
                "text": "UBRR sets the baud rate.", "page_start": 1, "page_end": 1}]


def fake_run_agent(doc_id, question, top_k, index_dir, client=None, model_name=None,
                    embed_model=None, verbose=True):
    return {
        "question": question, "answer": "UBRR = 51.",
        "retrieved_chunks": FAKE_CHUNKS,
        "tool_calls": [{"name": "calc_baud_rate_ubrr", "args": {}, "result": {"ubrr": 51}}],
    }


def fake_verify_answer(question, answer, chunks, tool_calls, client=None, model_name=None):
    return {"overall": "pass", "claims": [{"claim": "UBRR = 51", "verdict": "supported"}]}


ev.run_agent = fake_run_agent
ev.verify_answer = fake_verify_answer

# Write a tiny temp eval set + chunks file so run_eval's file loading path is
# exercised for real, not just the in-memory functions.
tmp_dir = Path(tempfile.gettempdir()) / "eval_mechanics_test"
tmp_dir.mkdir(parents=True, exist_ok=True)
(tmp_dir / "eval.jsonl").write_text(
    json.dumps({"id": "q01", "question": "UBRR for 9600 baud?",
                "relevant_keywords": ["UBRR", "baud"], "requires_tool": True,
                "expected_tool": "calc_baud_rate_ubrr"}) + "\n"
)
processed_dir = tmp_dir / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)
(processed_dir / "test_chunks.jsonl").write_text(json.dumps(FAKE_CHUNKS[0]) + "\n")
index_dir = tmp_dir / "index"
index_dir.mkdir(parents=True, exist_ok=True)

results = ev.run_eval("test", tmp_dir / "eval.jsonl", index_dir, client=object(), verbose=False)
check("run_eval returns one result per eval item", len(results), 1)
check("precision computed from the fake chunks file", results[0]["retrieval_precision"], 1.0)
check("faithfulness computed from fake verify_answer output", results[0]["faithfulness"], 1.0)
check("tool_used_correctly True (expected tool was in tool_calls_made)",
      results[0]["tool_used_correctly"], True)

print(f"\n{'ALL TESTS PASSED' if not failures else f'{len(failures)} FAILURE(S): ' + ', '.join(failures)}")
if failures:
    sys.exit(1)
