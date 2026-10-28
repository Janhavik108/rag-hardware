"""
Phase 5 - Self-verification
------------------------------
Second pass: checks each claim in the draft answer against the actual
retrieved evidence (chunks + any tool results) -- not by re-deriving the
answer, but by checking whether each specific claim is actually STATED in
the evidence it claims to be grounded in.

This is the "did the model's claim actually appear in the retrieved chunk"
check from the project brief -- the piece student RAG projects skip, and
the reason "how do you know it's not hallucinating" has an actual answer
here instead of a shrug.

Deliberately does NOT re-derive computed values (e.g. re-checking that
UBRR=51 is arithmetically correct) -- an LLM re-doing the same arithmetic
is exactly as unreliable as the LLM that did it the first time. What it CAN
check: whether a computed value came from the calculator tool (trusted,
deterministic) or was just asserted by the drafting model with no tool
backing it -- a real, meaningful signal even for answers that happen to be
numerically correct. (This distinction exists because of a real finding
while testing Phase 4: the agent answered an ADC prescaler question
correctly WITHOUT calling calc_adc_prescaler -- right answer, unverified
arithmetic. Verification should catch that pattern, not just outright
wrong answers.)

Usage:
    python src/verify.py --doc-id atmega32 --question "What UBRR value do I need for 9600 baud at 8MHz?"
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import get_llm_client, run_agent  # noqa: E402
from llm_utils import create_completion  # noqa: E402

VERIFY_SYSTEM_PROMPT = """You are a strict fact-checker reviewing a draft answer about a microcontroller datasheet.

You will be given:
1. Retrieved datasheet excerpts -- the ONLY allowed source of truth for factual/textual claims.
2. Tool computation results (if any) -- the ONLY allowed source of truth for derived numeric \
values like UBRR or prescaler settings.
3. A draft answer that claims to be grounded in the above.

Break the draft answer into its individual factual or numeric claims. For EACH claim, decide:
- "supported": directly stated in the excerpts, OR exactly matches a tool result.
- "unsupported": not found in the excerpts and not confirmed by a tool result. This INCLUDES \
numeric values the answer appears to have computed itself rather than sourcing from a tool, \
even if the arithmetic looks correct -- you are checking whether the SOURCE confirms the claim, \
not independently re-deriving it. You are not a calculator; you are a citation checker.
- "partially_supported": part of a compound claim is grounded, part isn't.

Respond with ONLY a JSON array, no other text, no markdown fences:
[{"claim": "...", "verdict": "supported|unsupported|partially_supported", "evidence": "chunk_id, tool name, or null"}]
"""


def format_evidence(chunks: list[dict], tool_calls: list[dict]) -> str:
    parts = ["=== Retrieved excerpts ==="]
    for c in chunks:
        parts.append(f"[{c['chunk_id']} | {c['section_title']}]\n{c['text']}")
    if tool_calls:
        parts.append("\n=== Tool results ===")
        for tc in tool_calls:
            parts.append(f"[{tc['name']}({tc['args']})] -> {json.dumps(tc['result'])}")
    return "\n\n".join(parts)


def _extract_json_array(text: str) -> list:
    """Models sometimes wrap JSON in markdown fences or add a stray sentence
    despite instructions not to -- pull out the first [...] block rather
    than failing outright on a strict json.loads()."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not parse a JSON array from verifier output: {text[:200]!r}")


def verify_answer(question: str, answer: str, chunks: list[dict], tool_calls: list[dict],
                   client=None, model_name: str = None) -> dict:
    """Runs the verification pass. Returns an overall verdict plus the
    per-claim breakdown the overall verdict was computed from.

    overall is one of:
      "pass"             -- every claim supported
      "flagged"          -- at least one claim unsupported
      "flagged_partial"  -- no outright-unsupported claims, but at least one partial
      "verification_failed" -- the verifier's own output couldn't be parsed;
                                treat this the same as "don't trust the draft"
                                rather than silently passing it through

    `client` can be injected for testing, same pattern as agent.run_agent().
    """
    client = client or get_llm_client()
    model_name = model_name or os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")

    evidence = format_evidence(chunks, tool_calls)
    user_prompt = (
        f"Question: {question}\n\nDraft answer to check:\n{answer}\n\n"
        f"Evidence available to the drafting model:\n\n{evidence}"
    )

    response = create_completion(
        client,
        model=model_name,
        messages=[
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = response.choices[0].message.content

    try:
        claims = _extract_json_array(raw)
    except (ValueError, json.JSONDecodeError) as e:
        return {"overall": "verification_failed", "claims": [], "error": str(e), "raw_response": raw}

    unsupported = [c for c in claims if c.get("verdict") == "unsupported"]
    partial = [c for c in claims if c.get("verdict") == "partially_supported"]

    if unsupported:
        overall = "flagged"
    elif partial:
        overall = "flagged_partial"
    else:
        overall = "pass"

    return {"overall": overall, "claims": claims, "raw_response": raw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--index-dir", default="data/index")
    args = ap.parse_args()

    result = run_agent(args.doc_id, args.question, args.top_k, Path(args.index_dir))
    print(f"\nQuestion: {result['question']}\n")
    print(f"Draft answer:\n{result['answer']}\n")

    verification = verify_answer(
        result["question"], result["answer"], result["retrieved_chunks"], result["tool_calls"],
    )

    print(f"Verification: {verification['overall'].upper()}\n")
    for c in verification.get("claims", []):
        print(f"  [{c.get('verdict', '?')}] {c.get('claim', '?')}  (evidence: {c.get('evidence', 'none')})")
    if verification["overall"] == "verification_failed":
        print(f"  (verifier output could not be parsed: {verification.get('error')})")

    if verification["overall"] in ("flagged", "flagged_partial", "verification_failed"):
        print("\nThis answer contains claims not directly supported by the retrieved evidence.")


if __name__ == "__main__":
    main()
