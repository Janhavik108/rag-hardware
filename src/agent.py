"""
Phase 4 - Agent loop
----------------------
retrieve -> draft (with tool use). Self-verification (checking the draft
against retrieved chunks) is Phase 5, not here -- this module's job ends at
producing a draft answer plus the evidence (chunks, tool calls) Phase 5 will
need to check it against.

Flow:
  1. Hybrid-retrieve top-k chunks for the question (reuses retrieve.py).
  2. Send the question + retrieved context to the LLM with the calculator
     tools (src/tools.py) available.
  3. If the model calls a tool, execute it locally and feed the result back
     for a final answer. If not, the model answers straight from context.

LLM access is any OpenAI-compatible chat completions endpoint -- Groq's free
tier, a local Ollama server, OpenAI itself -- configured via three env vars
(see .env.example). Nothing else in this file needs to change to switch
providers.

Usage:
    python src/agent.py --doc-id atmega32 --question "What UBRR value do I need for 9600 baud at 8MHz?"
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import (bm25_search, dense_search,  # noqa: E402
                       load_bm25, load_faiss, load_meta, reciprocal_rank_fusion)
from tools import TOOL_DISPATCH, TOOL_SCHEMAS  # noqa: E402
from llm_utils import create_completion  # noqa: E402

SYSTEM_PROMPT = """You are an assistant answering engineering questions about a microcontroller \
using ONLY the provided datasheet excerpts as your source of truth.

Rules:
- Base every factual claim on the excerpts given to you. If the excerpts don't contain \
the answer, say so explicitly instead of guessing.
- For any question involving arithmetic on register values, clock frequencies, baud \
rates, or prescalers, call the relevant tool instead of computing it yourself -- even \
if you think you know the formula. The tools exist specifically because hand-computed \
register math is a common source of subtly wrong answers.
- Cite which section/page the information came from when possible.
- Keep answers concise and technical -- this is for an engineer, not a tutorial."""


def retrieve_context(doc_id: str, question: str, top_k: int, index_dir: Path,
                      embed_model=None) -> list[dict]:
    """Hybrid retrieval -- same fusion logic as retrieve.py, just returning
    chunk dicts directly instead of printing them.

    `embed_model` can be a preloaded SentenceTransformer, so callers running
    many questions in a row (eval.py) don't reload the model every time --
    left as None here for single-question CLI use, where that doesn't matter.
    """
    meta = load_meta(doc_id, index_dir)
    bm25 = load_bm25(doc_id, index_dir)
    index = load_faiss(doc_id, index_dir)
    if embed_model is None:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    dense_results = dense_search(embed_model, index, question, k=20)
    bm25_results = bm25_search(bm25, question, k=20)
    fused = reciprocal_rank_fusion(dense_results, bm25_results)
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [meta[idx] for idx, _ in ranked]


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Excerpt {i} | {c['section_title']} | pages {c['page_start']}-{c['page_end']} "
            f"| {c['chunk_id']}]\n{c['text']}"
        )
    return "\n\n".join(parts)


def get_llm_client():
    """OpenAI-compatible client pointed at whatever LLM_API_BASE is configured.
    Works unmodified with Groq, a local Ollama server, OpenAI, etc."""
    from dotenv import load_dotenv
    load_dotenv()
    from openai import OpenAI

    base_url = os.environ.get("LLM_API_BASE")
    api_key = os.environ.get("LLM_API_KEY", "not-needed")  # Ollama ignores this
    if not base_url:
        raise SystemExit(
            "LLM_API_BASE is not set. Copy .env.example to .env and fill in your "
            "provider's details (Groq, Ollama, etc.) -- see README Phase 4 section."
        )
    return OpenAI(base_url=base_url, api_key=api_key)


def run_agent(doc_id: str, question: str, top_k: int = 5,
              index_dir: Path = Path("data/index"), client=None,
              model_name: str = None, verbose: bool = True, embed_model=None) -> dict:
    """Runs retrieve -> draft (+ tool use) once.

    Returns the draft answer plus everything Phase 5's verifier will need:
    the retrieved chunks and any tool calls made, so verification has real
    evidence to check the answer against rather than re-reading the same
    LLM output it's trying to check.

    `client` can be injected (e.g. a test double) to bypass get_llm_client();
    this is also how scripts/test_agent_mechanics.py exercises this function
    without a real API key or network access. `embed_model` can be a
    preloaded SentenceTransformer -- see retrieve_context().
    """
    chunks = retrieve_context(doc_id, question, top_k, index_dir, embed_model=embed_model)
    context = format_context(chunks)

    client = client or get_llm_client()
    model_name = model_name or os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Datasheet excerpts:\n\n{context}\n\nQuestion: {question}"},
    ]

    tool_calls_made = []
    response = create_completion(
        client, model=model_name, messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto",
    )
    msg = response.choices[0].message

    if msg.tool_calls:
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            args = json.loads(tc.function.arguments)
            if verbose:
                print(f"[agent] tool call: {fn_name}({args})")
            if fn_name not in TOOL_DISPATCH:
                result = {"error": f"unknown tool {fn_name}"}
            else:
                try:
                    result = TOOL_DISPATCH[fn_name](**args)
                except Exception as e:
                    result = {"error": str(e)}
            tool_calls_made.append({"name": fn_name, "args": args, "result": result})

        # Deliberately NOT replaying the raw assistant tool_calls message +
        # tool-role message back into a continued conversation here. Some
        # providers' models (observed with Groq's gpt-oss family) will keep
        # reasoning about further tool calls if the tool-call turn is still
        # in context -- even with tool_choice="none" -- and the API rejects
        # that outright rather than dropping it, crashing the whole request.
        # A fresh single-shot prompt that summarizes the tool result as plain
        # text has no tool-call-shaped turn for the model to "continue," and
        # is also more robust across providers generally, since it doesn't
        # depend on tool-message threading working the same way everywhere.
        tool_summary = "\n".join(
            f"- {tc['name']}({tc['args']}) returned: {json.dumps(tc['result'])}"
            for tc in tool_calls_made
        )
        followup_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Datasheet excerpts:\n\n{context}\n\nQuestion: {question}\n\n"
                f"The following calculation(s) were already run for you:\n{tool_summary}\n\n"
                f"Using these results and the excerpts above, give the final answer to the question."
            )},
        ]
        try:
            final = create_completion(client, model=model_name, messages=followup_messages)
            answer = final.choices[0].message.content
        except Exception as e:
            answer = (
                f"[Tool call succeeded but the follow-up answer request failed: {e}]\n"
                f"Raw tool result(s): {json.dumps([tc['result'] for tc in tool_calls_made])}"
            )
    else:
        answer = msg.content

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": chunks,
        "tool_calls": tool_calls_made,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--index-dir", default="data/index")
    args = ap.parse_args()

    result = run_agent(args.doc_id, args.question, args.top_k, Path(args.index_dir))

    print(f"\nQuestion: {result['question']}\n")
    if result["tool_calls"]:
        print("Tool calls:")
        for tc in result["tool_calls"]:
            print(f"  {tc['name']}({tc['args']}) -> {tc['result']}")
        print()
    print(f"Answer:\n{result['answer']}\n")
    print(f"Retrieved from: {[c['chunk_id'] for c in result['retrieved_chunks']]}")


if __name__ == "__main__":
    main()
