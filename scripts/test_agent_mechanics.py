"""
Sanity-checks the AGENT LOOP mechanics (tool dispatch, message construction,
the two-round tool-call -> final-answer flow, and the no-tool-call path)
using a mock LLM client instead of a real API. Tests the real src/agent.py
code, not a reimplementation -- only the two genuinely external dependencies
(the LLM client and hybrid retrieval) are faked, via the injection points
agent.run_agent() already exposes for exactly this purpose.

This does NOT test whether a real LLM actually calls tools correctly or
writes good answers -- only that the surrounding Python plumbing doesn't
crash and wires tool results back into the conversation correctly. Run
agent.py for real, with a real provider configured, for that.

Usage:
    python scripts/test_agent_mechanics.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import agent  # noqa: E402

failures = []


def check(label, actual, expected):
    ok = actual == expected
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: got {actual!r}, expected {expected!r}")
    if not ok:
        failures.append(label)


# ---------------------------------------------------------------------------
# Minimal fakes matching the slice of the OpenAI client surface agent.py uses
# ---------------------------------------------------------------------------

class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments  # JSON string, as the real SDK provides


class FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant"}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in self.tool_calls
            ]
        return d


class FakeResponse:
    def __init__(self, message):
        self.choices = [type("Choice", (), {"message": message})()]


class FakeClient:
    """Returns queued responses in order, one per .chat.completions.create() call."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

        class _Completions:
            def create(inner_self, **kwargs):
                self.calls.append(kwargs)
                return self._responses.pop(0)

        self.chat = type("Chat", (), {"completions": _Completions()})()


FAKE_CHUNKS = [
    {"chunk_id": "test_c0000", "section_title": "USART", "chunk_type": "prose",
     "page_start": 1, "page_end": 1, "text": "The baud rate is set by UBRR."},
]


def fake_retrieve_context(doc_id, question, top_k, index_dir, embed_model=None):
    return FAKE_CHUNKS


# ---------------------------------------------------------------------------
# Scenario 1: model calls a tool, then gives a final answer
# ---------------------------------------------------------------------------

print("=== Scenario: tool call path ===")
agent.retrieve_context = fake_retrieve_context  # monkeypatch the module-level function

tool_args = {"f_osc_hz": 8_000_000, "target_baud": 9600}
first_response = FakeResponse(FakeMessage(
    content=None,
    tool_calls=[FakeToolCall("call_1", "calc_baud_rate_ubrr", json.dumps(tool_args))],
))
second_response = FakeResponse(FakeMessage(content="Use UBRR=51 for 9600 baud at 8MHz."))
client = FakeClient([first_response, second_response])

result = agent.run_agent("test", "What UBRR for 9600 baud at 8MHz?", client=client, verbose=False)

check("two LLM calls made (draft + final)", len(client.calls), 2)
check("one tool call recorded", len(result["tool_calls"]), 1)
check("correct tool was called", result["tool_calls"][0]["name"], "calc_baud_rate_ubrr")
check("tool actually executed (real UBRR value in result)", result["tool_calls"][0]["result"]["ubrr"], 51)
check("final answer is the second response's content",
      result["answer"], "Use UBRR=51 for 9600 baud at 8MHz.")
check("retrieved chunks passed through", result["retrieved_chunks"], FAKE_CHUNKS)

# Second call should be a FRESH prompt (no replayed tool_calls/tool-role
# turns) with the tool result summarized as plain text in the user message.
second_call_messages = client.calls[1]["messages"]
tool_role_messages = [m for m in second_call_messages if m.get("role") == "tool"]
check("second call does NOT replay a tool-role message", len(tool_role_messages), 0)
check("second call is a fresh 2-message prompt (system + user)", len(second_call_messages), 2)
user_content = second_call_messages[1]["content"]
check("tool result (ubrr=51) appears as plain text in the follow-up prompt",
      '"ubrr": 51' in user_content, True)

# ---------------------------------------------------------------------------
# Scenario 2: model answers directly, no tool call needed
# ---------------------------------------------------------------------------

print("\n=== Scenario: no tool call path ===")
direct_response = FakeResponse(FakeMessage(content="The USART is described on page 1."))
client2 = FakeClient([direct_response])

result2 = agent.run_agent("test", "What is the USART?", client=client2, verbose=False)

check("only one LLM call made (no tool round)", len(client2.calls), 1)
check("no tool calls recorded", result2["tool_calls"], [])
check("answer is the direct response content", result2["answer"], "The USART is described on page 1.")

# ---------------------------------------------------------------------------
# Scenario 3: unknown tool name doesn't crash the loop
# ---------------------------------------------------------------------------

print("\n=== Scenario: unknown tool name handled gracefully ===")
bad_tool_response = FakeResponse(FakeMessage(
    content=None,
    tool_calls=[FakeToolCall("call_2", "nonexistent_tool", "{}")],
))
final_after_bad_tool = FakeResponse(FakeMessage(content="I couldn't compute that."))
client3 = FakeClient([bad_tool_response, final_after_bad_tool])

result3 = agent.run_agent("test", "Some question", client=client3, verbose=False)
check("unknown tool call recorded with an error result",
      "error" in result3["tool_calls"][0]["result"], True)
check("loop completes despite unknown tool", result3["answer"], "I couldn't compute that.")

print(f"\n{'ALL TESTS PASSED' if not failures else f'{len(failures)} FAILURE(S): ' + ', '.join(failures)}")
if failures:
    sys.exit(1)
