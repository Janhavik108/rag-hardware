# Findings

Concrete evidence gathered from real eval runs (a 22-question hand-labeled
set, `data/eval/atmega32_eval.jsonl`) against the ATmega32 datasheet
corpus, using Groq's `openai/gpt-oss-120b` (free tier). Every claim below
is backed by an actual transcript, not a hypothetical — that distinction
is the point of this document, in keeping with the project's own thesis:
don't assert something is true, show where it's grounded.

## 1. Self-verification catches true-but-ungrounded claims, not just hallucinations

The most important thing this project demonstrates isn't "the agent
sometimes makes things up" — it's that a claim can be **factually correct
and still unverified**, and that distinction is exactly what a citation
checker (as opposed to a fact checker) is for.

**Flagship example.** Asked *"What ADC prescaler should I use at 16MHz to
stay in the recommended range?"*, the agent answered ÷128, 125kHz, correctly
— but never called `calc_adc_prescaler`. It did the division itself. The
verifier caught this:

```
[unsupported] With a 16 MHz CPU clock, the ADC clock becomes 16MHz / 128 = 125 kHz. (evidence: None)
```

Right answer. No traceable source. Flagged anyway — on purpose. An LLM
re-deriving arithmetic it already got right once isn't evidence the
arithmetic is reliable; it's the same unverified step happening twice.

**This exact case reproduced a second time, independently, in the full
22-question run** — same question, same skipped tool, same computed
125kHz claim, flagged `unsupported` again. But this time faithfulness
dropped to **0.2**, not just one claim: the model's entire justification
chain ("higher-division settings exceed the limit," "/128 is the correct
choice") got marked unsupported along with the arithmetic, because none of
it traced back to anything except the model's own unverified reasoning.
Skipping the tool didn't just leave one number ungrounded — it left the
whole argument ungrounded. Two independent runs, same question, same
failure shape, worse the second time. That's a pattern, not noise.

**Confirmed pattern, not a one-off.** The same shape of failure showed up
independently across three separate runs of a RESET-pin question, each time
eliciting a different mix of directly-cited facts (pulse width, threshold
voltage, timeout — reliably `supported`) plus true AVR domain knowledge the
model added from general training rather than the retrieved excerpts:
"internal registers are cleared," "the pins have no defined logic level,"
"peripherals are cleared or placed in their reset condition." All correct.
None traceable to the specific chunks retrieved for that question. All
flagged. That's a repeatable behavior pattern, not noise.

## 2. A three-category framework for grounding failures

Investigating *why* the retrieved context didn't cover a true claim
surfaced three genuinely different failure modes, confirmed against the
actual corpus rather than assumed:

1. **Hallucination** — the claim isn't grounded anywhere in the document.
2. **Retrieval miss** — the claim *is* grounded, in a specific real chunk,
   but that chunk didn't make the top-k for this query. Confirmed case: the
   fact "Port pins are tri-stated when a reset condition becomes active" is
   stated verbatim in the Port A–D chunks, but a query about "the RESET pin"
   shares almost no vocabulary with "tri-stated" or "Port A" — neither BM25
   nor dense retrieval surfaced it at `top_k=5`.
3. **Structurally unretrievable** — the claim is true and technically
   documented, but only as an aggregate pattern (e.g. the "Initial Value"
   row present in nearly every register's bit-table throughout the
   datasheet) rather than as a single declarative sentence anywhere. No
   retrieval strategy at any reasonable `top_k` synthesizes "registers reset
   to defaults" from 50+ scattered per-register table rows — that requires
   generalization the source document itself never performs in one place.

Category 3 is the most interesting for an interview: it's not a retrieval
*bug*, it's a structural limit of chunk-based RAG against documents that
encode some facts as repeated structure instead of prose.

## 3. Tool-use compliance isn't guaranteed by prompting alone — and the metric alone doesn't tell the whole story either

The system prompt explicitly instructs the model to call the calculator
tools rather than compute register math itself. The full 22-question eval
run measured `tool_use_compliance_rate: 0.5` — 2 of 4 tool-required
questions actually called their tool. That's not a category-specific gap
(it's not "ADC always skips it" or "baud always skips it"): one baud
question called the tool and one didn't; one ADC question called the tool
and one didn't. It looks like inconsistent, close-to-coin-flip behavior
across otherwise identically-shaped questions, not a systematic pattern
tied to one calculation type.

**The more interesting finding is what the two "non-compliant" cases
actually looked like when inspected**, because they were completely
different in a way the compliance metric alone can't distinguish:

- One (`"What ADC prescaler should I use at 16MHz..."`) is the ADC case
  from §1 above: skipped the tool, computed the value itself, faithfulness
  dropped to 0.2. A real failure.
- The other (`"What UBRR value is needed for 4800 baud at 1MHz using
  double speed mode?"`) also skipped the tool — but faithfulness came back
  a perfect **1.0**. Why: the retrieved context included a datasheet table
  ("Examples of Baud Rate Setting") that directly tabulates `UBRR=25` for
  exactly this baud rate and oscillator frequency, and the model correctly
  cited the table row instead of computing anything. Not a failure at
  all — arguably the *better* behavior, since it preferred a directly-
  stated fact over redundant arithmetic.

`tool_use_compliance_rate` alone reports both of these as identical
failures (0 for 0 — tool not called). Only `faithfulness` (0.2 vs. 1.0)
reveals they're opposite outcomes. **Neither metric alone is sufficient;
the combination is what actually tells you whether skipping the tool was a
problem.** That's a more useful thing to say in an interview than "the
agent sometimes skips tools" — it's "a single compliance metric is the
wrong level of granularity for this question, and here's the second metric
that resolves the ambiguity, with a concrete example of each case."

n=4 tool-required questions total is still a small sample — worth treating
this as a real, reproduced pattern (inconsistent tool use, sometimes
harmless, sometimes not) rather than a precise rate.

## 4. Verification itself has confirmed, specific limitations

- **Citation mis-attribution.** In one run, the verifier correctly marked a
  "< 0.5% error" claim as `supported` — but cited `calc_baud_rate_ubrr`
  (which never mentions 0.5%) instead of the real source, a datasheet
  table's bold-value legend. The *verdict* was right; the *evidence field*
  pointed at the wrong place. Single-pass verification can get supported/
  unsupported correct while still mis-attributing which piece of context
  backs it, when more than one plausible source is present.
- **Refusal-heuristic gap, confirmed on both refusal-test questions in the
  full run (2/2, not a one-off).** `eval.py`'s rough refusal detector missed
  both real refusals: one said *"...cannot be **determined** from the
  information given"* (passive voice — the phrase list had "cannot
  **determine**," active voice, a different string), the other said
  *"...is not **stated** in the supplied information"* (the list had "not
  **found** in," a different verb). Both answers were genuinely, almost
  perfectly grounded refusals — one scored a full 1.0 faithfulness, the
  other 0.83 with only a minor verifier nitpick on a summary sentence — so
  the underlying safety behavior worked correctly both times. Only the
  crude keyword convenience-heuristic measuring it failed, twice,
  independently, on two different phrasing variations. Confirms the
  README's own prediction about this heuristic rather than just repeating
  it: it's brittle to phrasing, and expanding the phrase list to patch
  these two specific misses wouldn't fix the underlying brittleness, just
  move where the next miss happens.

## 5. Eval methodology lessons (the meta-findings)

- **Keyword ground truth needs per-question care.** Copying the same
  `relevant_keywords` between two similarly-worded questions (RESET-pin
  behavior vs. I/O-pin-during-reset) produced a false `0.0` precision/recall
  on a question with genuinely excellent, correctly-cited retrieval —
  because the copied keywords matched the *other* question's evidence, not
  this one's. Caught and fixed by inspecting the actual retrieved chunks
  rather than trusting the metric at face value. The lesson: write ground
  truth against what each specific question needs, never copy-paste it.
- **Free-tier infrastructure is a real, practical constraint — and it's
  more than one kind of limit.** First hit a per-minute token cap (TPM) on
  a 7-question run, even with proactive delay between questions and
  6-attempt exponential backoff. Fixed by increasing the delay — but the
  full 22-question run later hit a *different*, harder wall: a **daily**
  token quota (TPD), the cumulative total across everything run on the
  account that day, not just one eval invocation. No amount of in-process
  retrying fixes a daily cap — waiting a few minutes doesn't help if the
  day's budget is actually gone. That distinction (per-minute vs.
  cumulative-daily) isn't obvious from a single rate-limit error and only
  showed up by actually hitting both kinds live. The real fix wasn't more
  patience, it was building `eval.py --resume`: incremental checkpointing
  after every question, so a run interrupted at question 15 of 22 doesn't
  discard the 14 already-completed results — the next invocation, whenever
  quota allows (a different account, the next day, whatever), picks up
  exactly where it stopped. That's a better engineering response to a real
  constraint than either working around it with a second account or
  padding the delay further and hoping.

## Bottom line

Measured from a completed 22-question eval run: mean retrieval precision
0.42, mean retrieval recall 0.54, mean answer faithfulness 0.82,
verification flagged rate 0.55, tool-use compliance 0.5 (n=4), refusal-
heuristic rate 0.0 (n=2, both confirmed false negatives, not real
failures — see §4). Those numbers matter less on their own than what
produced them: the same true-but-ungrounded-claim pattern reproduced twice
independently on the same question, a case where skipping a tool was
actually fine and only a second metric could tell the difference, and two
independent, confirmed limitations in the verification and eval tooling
itself, found by checking rather than assuming either was working. That's
the actual answer to "how do you know it's not hallucinating" — not a
claim that it doesn't, but a demonstrated, repeatable way of finding out
when it might, including the places that method itself still needs a
human glancing at the transcript rather than trusting the number alone.
