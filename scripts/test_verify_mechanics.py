"""
Sanity-checks verify.py: overall-verdict aggregation logic, robustness of
JSON extraction against models that don't follow the "ONLY JSON" instruction
perfectly, and evidence formatting -- using a mock LLM client, no API key or
network needed.

Usage:
    python scripts/test_verify_mechanics.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import verify  # noqa: E402

failures = []


def check(label, actual, expected):
    ok = actual == expected
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: got {actual!r}, expected {expected!r}")
    if not ok:
        failures.append(label)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeResponse:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": FakeMessage(content)})()]


class FakeClient:
    def __init__(self, content):
        self._content = content
        self.last_call_kwargs = None

        class _Completions:
            def create(inner_self, **kwargs):
                self.last_call_kwargs = kwargs
                return FakeResponse(self._content)

        self.chat = type("Chat", (), {"completions": _Completions()})()


FAKE_CHUNKS = [{"chunk_id": "c1", "section_title": "USART", "text": "UBRR = f_osc/(16*baud) - 1"}]
FAKE_TOOL_CALLS = [{"name": "calc_baud_rate_ubrr",
                     "args": {"f_osc_hz": 8000000, "target_baud": 9600},
                     "result": {"ubrr": 51}}]

print("=== _extract_json_array robustness ===")

check("parses clean JSON", verify._extract_json_array('[{"claim": "x", "verdict": "supported"}]'),
      [{"claim": "x", "verdict": "supported"}])

check("parses JSON wrapped in markdown fences",
      verify._extract_json_array('```json\n[{"claim": "x", "verdict": "supported"}]\n```'),
      [{"claim": "x", "verdict": "supported"}])

check("parses JSON with a stray leading sentence",
      verify._extract_json_array('Here is my analysis:\n[{"claim": "x", "verdict": "supported"}]'),
      [{"claim": "x", "verdict": "supported"}])

try:
    verify._extract_json_array("This is not JSON at all, sorry.")
    check("raises on totally unparseable text", "no exception", "exception")
except (ValueError, json.JSONDecodeError):
    check("raises on totally unparseable text", "exception", "exception")

print("\n=== overall-verdict aggregation ===")

# All supported -> pass
client = FakeClient(json.dumps([
    {"claim": "UBRR is 51", "verdict": "supported", "evidence": "calc_baud_rate_ubrr"},
    {"claim": "8MHz oscillator", "verdict": "supported", "evidence": "c1"},
]))
r = verify.verify_answer("q", "a", FAKE_CHUNKS, FAKE_TOOL_CALLS, client=client)
check("all supported -> overall pass", r["overall"], "pass")
check("claims list has 2 entries", len(r["claims"]), 2)

# One unsupported -> flagged (this is the case that matters most: a computed
# value asserted without tool backing should land here)
client = FakeClient(json.dumps([
    {"claim": "UBRR is 51", "verdict": "supported", "evidence": "calc_baud_rate_ubrr"},
    {"claim": "ADC clock is 125kHz", "verdict": "unsupported", "evidence": None},
]))
r = verify.verify_answer("q", "a", FAKE_CHUNKS, [], client=client)
check("one unsupported -> overall flagged", r["overall"], "flagged")

# Only partial, no outright unsupported -> flagged_partial
client = FakeClient(json.dumps([
    {"claim": "Some compound claim", "verdict": "partially_supported", "evidence": "c1"},
]))
r = verify.verify_answer("q", "a", FAKE_CHUNKS, [], client=client)
check("only partial -> overall flagged_partial", r["overall"], "flagged_partial")

# Verifier output that can't be parsed at all -> verification_failed, not a
# silent pass (a broken verifier should never look like a clean bill of health)
client = FakeClient("I refuse to answer in JSON today.")
r = verify.verify_answer("q", "a", FAKE_CHUNKS, [], client=client)
check("unparseable verifier output -> verification_failed, not pass", r["overall"], "verification_failed")
check("verification_failed still carries raw_response for debugging",
      "raw_response" in r, True)

print("\n=== format_evidence ===")
evidence = verify.format_evidence(FAKE_CHUNKS, FAKE_TOOL_CALLS)
check("evidence includes chunk id", "c1" in evidence, True)
check("evidence includes chunk text", "UBRR = f_osc" in evidence, True)
check("evidence includes tool name", "calc_baud_rate_ubrr" in evidence, True)
check("evidence includes tool result", "51" in evidence, True)

evidence_no_tools = verify.format_evidence(FAKE_CHUNKS, [])
check("no '=== Tool results ===' section when no tools were called",
      "Tool results" in evidence_no_tools, False)

print(f"\n{'ALL TESTS PASSED' if not failures else f'{len(failures)} FAILURE(S): ' + ', '.join(failures)}")
if failures:
    sys.exit(1)
