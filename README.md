# Agentic RAG over Hardware Datasheets

Agent that answers engineering questions from microcontroller/IC datasheets
using hybrid retrieval, tool use, and a retrieve → draft → verify loop.

**→ See [FINDINGS.md](FINDINGS.md) for the actual evidence this project produced** —
confirmed cases of correct-but-ungrounded answers, a three-category framework
for grounding failures, and honest limitations of the verifier itself. That
document is the real deliverable; this README is the how-to-run reference.

## Architecture

```
                    ┌─────────────┐
   datasheet PDF →  │  ingest.py  │  → text spans + tables (font-aware)
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │chunk_docs.py│  → section-level chunks (prose/table)
                    └──────┬──────┘
                           ↓
              ┌────────────┴────────────┐
              ↓                         ↓
      ┌───────────────┐         ┌───────────────┐
      │  FAISS (dense) │         │  BM25 (sparse) │   build_index.py
      └───────┬───────┘         └───────┬───────┘
              └────────────┬────────────┘
                            ↓
                    ┌───────────────┐
        query   →   │  retrieve.py  │  → Reciprocal Rank Fusion
                    └───────┬───────┘
                            ↓
                    ┌───────────────┐      ┌──────────────┐
                    │   agent.py    │ ←──→ │  tools.py     │
                    │ (draft + tool │      │ (UBRR calc,   │
                    │     use)      │      │  ADC prescaler)│
                    └───────┬───────┘      └──────────────┘
                            ↓
                    ┌───────────────┐
                    │   verify.py   │  → per-claim supported/
                    │ (2nd LLM pass)│    unsupported/partial
                    └───────┬───────┘
                            ↓
                    ┌───────────────┐
                    │    eval.py    │  → precision/recall,
                    │ (hand-labeled │    faithfulness,
                    │   question set)│   tool compliance
                    └───────────────┘
```

## Status
- [x] Phase 1: project scaffolding
- [x] Phase 2: ingestion + section/table-aware chunking
- [x] Phase 3: hybrid dense + BM25 retrieval
- [x] Phase 4: agent loop + tool use
- [x] Phase 5: self-verification
- [x] Phase 6: evaluation harness
- [x] Phase 7: polish

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Phase 1-2 pipeline

1. **Get a real datasheet.** Drop a PDF into `data/raw/`, e.g.
   `data/raw/atmega32.pdf` (search "ATmega32 datasheet Microchip PDF").

   Don't have one yet? Sanity-check the pipeline first with a synthetic one:
   ```bash
   python scripts/make_test_pdf.py
   ```

2. **Ingest** — extract text spans (with font metadata) and tables:
   ```bash
   python src/ingest.py --pdf data/raw/atmega32.pdf --doc-id atmega32
   ```
   Produces `data/processed/atmega32_pages.json` and `atmega32_tables.json`.

3. **Chunk** — group into section-level chunks, tables kept separate:
   ```bash
   python src/chunk_docs.py --doc-id atmega32
   ```
   Produces `data/processed/atmega32_chunks.jsonl` — one JSON object per
   line: `chunk_id`, `doc_id`, `chunk_type` (`prose`/`table`), `section_title`,
   `page_start`, `page_end`, `text`.

4. **Eyeball the output** — use the viewer instead of hand-rolling one-liners:
   ```bash
   python scripts/view_chunks.py --doc-id atmega32
   python scripts/view_chunks.py --doc-id atmega32 --type table
   python scripts/view_chunks.py --doc-id atmega32 --section usart
   python scripts/view_chunks.py --doc-id atmega32 --id atmega32_c0012 --full
   ```
   (Windows: use `py` instead of `python`.)

   Check that:
   - Section titles look like real headings, not sentence fragments or a
     wall of merged revision-history bullets.
   - Table chunks aren't duplicated inside prose chunks, and aren't empty
     boxes (decorative diagrams/pinouts) mistaken for real tables.
   - No huge chunk is swallowing multiple unrelated sections (means a
     heading was missed — check the font size it's using vs. body text).

   Real datasheets have boilerplate that isn't worth answering questions
   from (revision history, errata, ordering/packaging info). These sections
   are usually made of many small sub-items (e.g. each errata entry, each
   changelog line) whose own titles don't repeat the parent heading, so a
   plain keyword match on `--exclude-sections` alone won't catch them. Add
   `--exclude-tail` to cut everything from the first match to the end of
   the document — safe here because these sections cluster at the end:
   ```bash
   python src/chunk_docs.py --doc-id atmega32 --exclude-tail \
       --exclude-sections "revision history,errata,ordering information,packaging information"
   ```
   Check the printed cutoff page against the source PDF once, since
   `--exclude-tail` assumes nothing worth keeping comes after the first
   match — if your datasheet interleaves these sections with real content,
   skip the flag and rely on `--exclude-sections` alone (accepting that
   individual sub-items may slip through).

   If it's misbehaving on your specific datasheet's layout, the knobs to
   tune are in `src/chunk_docs.py`:
   - `is_heading()` — the font-size/bold threshold for "this line is a heading"
   - `MAX_HEADING_CHARS` — cap on merged multi-line headings (prevents a run
     of bold list items from snowballing into one giant unreadable title)
   - `strip_repeated_boilerplate()` — how aggressively running headers/footers
     get stripped (currently: short line, repeats on ≥20% of pages)
   - `strip_page_numbers()` — regex for "this line is just a page number";
     widen/narrow the digit-count cap if it's over- or under-matching
   - `_REGISTER_HEADER_RE` — a document-specific structural heading rule
     (see below) for datasheets that number individual registers in a
     consistent, greppable way
   - `--exclude-sections` / `--min-chars` — post-hoc filtering

### Generalization test: a second, structurally different datasheet

Ran the same pipeline against the ESP32 Technical Reference Manual (750+
pages, Espressif — a completely different chip family and PDF generator
than the ATmega32's Atmel-produced document) to check whether the chunking
heuristics generalize or were accidentally overfit to one document's font
choices. Two things confirmed the pipeline generalizes well without
changes: `strip_page_numbers()` and `strip_repeated_boilerplate()` worked
immediately (the latter even caught `"(reserved)"` — a bit-diagram label
repeated across hundreds of register tables — something that never came up
in the ATmega32 corpus, with zero tuning).

One thing didn't generalize, and needed a real fix: the ESP32 TRM numbers
every individual register with a highly consistent header —
`"Register 1.1. SENS_ULP_CP_SLEEP_CYCn_REG (0x18+0x4*n)"` — rendered at a
font size/weight that didn't clear `is_heading()`'s threshold (which was
tuned against the ATmega32's different font conventions). Result: 50+
consecutive registers collapsed under one stale chapter-level heading
("Registers"), instead of getting their own chunk each — exactly the
granularity a tool-building task needs. Fixed with a dedicated regex rule
(`_REGISTER_HEADER_RE`) checked independently of font size, added
alongside the existing heuristic rather than replacing it, since the
existing chapter-heading detection was working correctly and a global
threshold change risked reintroducing the false-positive problems already
solved for the ATmega32 corpus. Verified against both a synthetic
reproduction of the exact failure and a false-positive check against
plausible body prose ("See Register 1.1 for...") before shipping it.

**Lesson for the writeup:** a heuristic tuned against one document's
formatting can silently fail on a structurally different one without
crashing or erroring — it just produces worse chunks, which only shows up
by actually inspecting output on new input, not by assuming success. The
fix that worked here is also a useful general pattern: when a font-size
heuristic misses a real, document-specific structural marker, a narrow
regex rule ORed alongside it is safer than loosening the general threshold.

## Design notes (for the writeup / interview)
- **Why section-based chunking, not fixed windows?** Datasheets are already
  structurally segmented (register descriptions, timing sections, pin
  tables) — chunking on that structure keeps a chunk's content coherent,
  which matters a lot once retrieval + verification depend on "is this
  claim actually in this chunk."
- **Why extract tables separately from prose?** Register maps and pin
  tables are dense, exact-value content. Flattening them into paragraph
  text loses row/column alignment; keeping them as their own markdown-
  rendered chunks preserves the structure that later retrieval (Phase 3)
  needs to match exact register names.
- **Known limitation:** heading detection is font-size heuristic, not a
  real document outline. Good enough for a single-datasheet student
  project; would need per-document calibration (or an actual PDF outline
  when the file has bookmarks) to generalize across many datasheets. One
  specific edge case: if a page's only content is a heading plus a
  diagram/table (no prose), that heading has nothing to attach to and can
  bleed into the next section until `MAX_HEADING_CHARS` forces a break —
  rare on dense datasheet pages, but worth knowing about if a section title
  looks like two headings mashed together.

## Phase 3: hybrid retrieval

1. **Install the retrieval deps** (already in `requirements.txt`):
   ```bash
   pip install sentence-transformers faiss-cpu rank-bm25
   ```

2. **Build the index** — embeds every chunk with `sentence-transformers`
   (`all-MiniLM-L6-v2`) into a FAISS flat index, and builds a BM25 index
   over the same chunks:
   ```bash
   python src/build_index.py --doc-id atmega32
   ```
   First run downloads the embedding model (~90MB) from Hugging Face, so
   it needs normal internet access. Writes `data/index/atmega32.faiss`,
   `atmega32_bm25_corpus.pkl`, and `atmega32_meta.json`.

   No internet / just want the BM25 side working? `--skip-dense` builds
   the BM25 index and metadata without touching the embedding model at all
   — useful for a quick sanity check before committing to the full build:
   ```bash
   python src/build_index.py --doc-id atmega32 --skip-dense
   ```

3. **Query it** — fuses dense + BM25 rankings via Reciprocal Rank Fusion:
   ```bash
   python src/retrieve.py --doc-id atmega32 --query "how is the baud rate calculated"
   python src/retrieve.py --doc-id atmega32 --query "UBRR11 bit"
   ```
   Each result line shows `dense_rank` and `bm25_rank` alongside the fused
   score, so you can see which retriever actually surfaced it — useful for
   sanity-checking that hybrid is pulling its weight. Compare against
   either retriever alone:
   ```bash
   python src/retrieve.py --doc-id atmega32 --query "..." --dense-only
   python src/retrieve.py --doc-id atmega32 --query "..." --bm25-only
   ```
   Try a register-name query (e.g. `"ADPS2:0"`) and a purely conceptual one
   (e.g. `"how do I slow down the ADC clock"`) with `--dense-only` vs.
   `--bm25-only` — this is the concrete evidence for "why hybrid" that's
   worth having ready for an interview, and it's exactly what Phase 6's
   eval harness will measure formally instead of just eyeballing.

4. **Quick mechanics check without a model download** — validates FAISS
   save/load, BM25 scoring, and the RRF fusion math using fake embeddings.
   Doesn't test embedding *quality*, just that the plumbing works:
   ```bash
   python scripts/test_retrieval_mechanics.py --doc-id atmega32
   ```

### Design notes (Phase 3)
- **Why Reciprocal Rank Fusion instead of averaging scores?** Cosine
  similarity (bounded, [-1, 1]) and BM25 (unbounded, corpus-dependent) live
  on incomparable scales — averaging them directly would let whichever
  score happens to have larger magnitude dominate. RRF only uses each
  retriever's *rank*, sidestepping the normalization problem entirely.
  `rrf_k=60` (in `retrieve.py`) is the standard default from the original
  RRF paper; lower values weight top-ranked results more heavily.
- **Why embed `"{section_title}: {text}"` instead of just `text`?** A table
  chunk's raw text can be just `"| UBRRH | - | - | ... |"` with nothing
  indicating what it's a table *of* — the section title carries that
  context into the embedding.
- **Known limitation:** `IndexFlatIP` is exact (brute-force) nearest-
  neighbor search, which is fine at this corpus size (hundreds to low
  thousands of chunks) but doesn't scale to a large multi-datasheet corpus
  — that would need an approximate index (e.g. FAISS `IndexIVFFlat` or
  `IndexHNSWFlat`) instead.

## Phase 4: agent loop + tool use

1. **Install** (already in `requirements.txt`): `pip install openai python-dotenv`

2. **Configure an LLM provider.** `agent.py` talks to any OpenAI-compatible
   chat completions endpoint — Groq's free tier, a local Ollama server, or
   OpenAI itself — via three env vars:
   ```bash
   cp .env.example .env
   # edit .env: fill in LLM_API_BASE / LLM_API_KEY / LLM_MODEL
   ```
   `.env.example` has ready-to-use blocks for Groq (free, fast, supports
   tool calling), local Ollama (no API key, no internet needed after the
   model pull), and OpenAI. Tool-calling support varies by provider/model —
   Groq's larger models and Ollama's llama3.1+/qwen2.5/mistral-nemo support
   it; if your model doesn't, the agent still works, it just never calls
   `calc_baud_rate_ubrr`/`calc_adc_prescaler` and answers from context alone.

3. **Run it:**
   ```bash
   python src/agent.py --doc-id atmega32 --question "What UBRR value do I need for 9600 baud at 8MHz?"
   python src/agent.py --doc-id atmega32 --question "What prescaler should I use for the ADC at 16MHz?"
   ```
   Output shows any tool calls made (name, arguments, computed result),
   then the final answer, then which chunks it was grounded in.

4. **Test the loop mechanics without an API key or network access** — this
   is what I validated on my end, since I couldn't reach a live LLM
   provider from this sandbox:
   ```bash
   python scripts/test_agent_mechanics.py
   ```
   It runs the real `agent.run_agent()` against a scripted fake LLM client
   (tool-call round → final answer, direct-answer-no-tool-call, and an
   unknown-tool-name error case) and checks the tool dispatch, message
   construction, and return values are all correct. It does **not** test
   whether a real model actually chooses to call tools well — only that the
   surrounding code doesn't break when it does. Run `test_tools.py` too if
   you haven't (Phase 4 calculator unit tests, checked against manually
   computed reference values, not against memory of "the right answer").

### Design notes (Phase 4)
- **Why calculator tools instead of trusting the LLM's arithmetic?** Register
  math (baud rate divisors, ADC prescalers) is exactly the kind of thing
  LLMs get subtly, confidently wrong — off-by-one on the UBRR formula is a
  classic failure mode. The tools are deterministic Python; the system
  prompt explicitly tells the model to call them rather than compute by
  hand, which is a real, defensible design decision to discuss in an
  interview, not just "I added function calling because the assignment
  asked for an agent."
- **Why return `error_percent`/`warning` from the tools, not just the raw
  register value?** "What UBRR do I use" is an incomplete answer without
  knowing how far off the actual baud rate lands — the datasheet's own
  Asynchronous Operational Range table treats >2% error as a real risk of
  framing errors, so the tool surfaces that instead of silently returning a
  bad pick.
- **Why OpenAI-compatible via env vars instead of hardcoding one provider's
  SDK?** Keeps the agent loop itself provider-agnostic — swapping from
  Groq to a local Ollama model for offline testing is a `.env` edit, not a
  code change.
- **Known limitation:** the loop supports one round of tool calls followed
  by one final answer — it doesn't handle a model that wants to call a
  tool, look at the result, then call a *different* tool before answering.
  Fine for the two calculators here (there's no reason a single question
  needs both), but would need a proper loop (call until the model stops
  requesting tools) to generalize further.
- **Real bug found while testing against Groq:** the first version of this
  loop replayed the raw assistant tool-call turn back into the conversation
  before asking for a final answer (the standard OpenAI multi-turn tool
  pattern). Against Groq's `gpt-oss` models, that put the model in a state
  where it kept wanting to reason about *further* tool calls — even with
  `tool_choice="none"` explicitly set — and Groq's API rejected that
  outright instead of just dropping it, crashing the whole request. Fix:
  the follow-up call is now a fresh, single-shot prompt with the tool
  result summarized as plain text, not a continuation of the tool-call
  turn. No tool-call-shaped turn in context means nothing for the model to
  "continue," and it's more robust across providers generally since it
  doesn't depend on tool-message threading behaving identically everywhere.

## Phase 5: self-verification

1. **Run it** — wraps `agent.run_agent()`, then checks the draft against its
   own retrieved evidence:
   ```bash
   python src/verify.py --doc-id atmega32 --question "What UBRR value do I need for 9600 baud at 8MHz?"
   ```
   Output: the draft answer, an overall verdict (`PASS` / `FLAGGED` /
   `FLAGGED_PARTIAL` / `VERIFICATION_FAILED`), and a per-claim breakdown
   showing which specific claims were/weren't found in the evidence.

2. **Test the mechanics without an API key**, same pattern as Phases 3-4:
   ```bash
   python scripts/test_verify_mechanics.py
   ```
   Covers the overall-verdict aggregation logic, JSON-parsing robustness
   (models don't always follow "respond with ONLY JSON" perfectly — markdown
   fences and stray sentences are handled), and evidence formatting. Doesn't
   test whether a real model checks claims *well* — only that the surrounding
   code is correct.

### Design notes (Phase 5)
- **Why one combined "extract + verify" call instead of two passes?** Fewer
  API calls matters on a free tier, and there's no real accuracy loss —
  the model has to identify claims either way, so asking it to verify each
  one in the same pass it identifies them is strictly less wasteful than a
  separate extraction round-trip first.
- **Why does the verifier explicitly refuse to re-check arithmetic?** This
  is the most important design decision in this phase, and it came directly
  from a real finding in Phase 4 testing: the agent answered an ADC
  prescaler question *correctly* without ever calling `calc_adc_prescaler` —
  it just computed 16MHz ÷ 128 = 125kHz itself. If the verifier were allowed
  to independently check "is this arithmetic right," it would have rubber-
  stamped that answer, because the math happened to be correct. But an LLM
  re-deriving the same calculation is exactly as unreliable as the LLM that
  derived it the first time — checking it with more LLM inference doesn't
  add any real assurance. So the verifier is restricted to checking
  *provenance*, not correctness: is this specific value stated in a
  retrieved excerpt, or does it exactly match a tool's output? A correct-
  but-tool-less computed value gets flagged as `unsupported`, on purpose.
  That's not a false positive — it's the whole point. The answer being
  right this time doesn't mean the pattern is safe; an unverified LLM
  computation is a coin flip that happened to land heads.
- **Why `verification_failed` instead of defaulting to `pass` when the
  verifier's own output can't be parsed?** A verification step that fails
  open (silently trusts the draft when it can't produce a real verdict) is
  worse than no verification step at all — it looks like a safety net while
  providing none. Failing to `verification_failed` means a broken verifier
  is visible and treated with the same suspicion as a flagged answer,
  rather than silently waved through.
- **Known limitation:** this only *flags* unsupported claims — it doesn't
  retry retrieval with a reformulated query or attempt a self-correction
  loop, both of which were mentioned as options in the original project
  scope. Flagging is the right MVP: it's the mechanism that actually
  answers "how do you know it's not hallucinating," and retry logic is a
  reasonable Phase 6+ extension once you have eval numbers showing how
  often flagging actually fires and on what kinds of questions.
- **Confirmed finding (not hypothetical):** tested two real cases side by
  side. (1) An ADC prescaler question where the agent computed 125kHz
  itself without calling `calc_adc_prescaler` — the verifier correctly
  flagged that value as `unsupported` despite it being numerically right,
  exactly the behavior this phase exists to produce. (2) A baud rate
  question where the verifier marked a "< 0.5% error" claim as `supported`
  but cited the wrong evidence (`calc_baud_rate_ubrr`, which never mentions
  0.5%, instead of the actual source — a real datasheet table's bold-value
  legend). The *verdict* was right; the *citation* was wrong. That's a
  narrower, real bug: single-pass verification can get the supported/
  unsupported call correct while still mis-attributing which piece of
  evidence backs it, when multiple plausible sources are in context. Worth
  tracking as two separate metrics in Phase 6, not one: verdict accuracy
  and citation accuracy are different questions.


## Phase 6: evaluation harness

1. **The starter eval set** (`data/eval/atmega32_eval.jsonl`) has 6 example
   questions covering the topics we've actually confirmed are in your
   corpus, as real examples of the format — **this is not the 20-30
   question set the project scope calls for.** Writing the rest is on you,
   deliberately: the credibility of an eval harness comes from the ground
   truth being hand-verified against your actual datasheet by the person
   claiming the numbers, not generated by whoever wrote the code. Each line
   is one JSON object:
   ```json
   {"id": "q07", "question": "...", "relevant_keywords": ["TWI", "prescaler"], "requires_tool": false}
   ```
   - `relevant_keywords`: words that should ALL appear (case-insensitive) in
     a chunk's title+text for that chunk to count as relevant. Used for
     retrieval precision/recall — **not exact chunk IDs**, because chunk IDs
     shift every time `chunk_docs.py` reruns with different flags or a
     re-ingested PDF; keywords don't.
   - `requires_tool` / `expected_tool` (optional): set these for questions
     that should trigger `calc_baud_rate_ubrr` or `calc_adc_prescaler`, to
     measure tool-use compliance — whether the agent actually reaches for
     the tool it has, which Phase 4/5 testing already showed isn't
     guaranteed just because the system prompt asks for it.
   - `expect_refusal` (optional): set for questions the datasheet genuinely
     can't answer, to spot-check that the agent says so instead of
     fabricating (checked with a rough keyword heuristic — read the actual
     answer yourself too, don't trust the heuristic alone).
   - Aim for a real mix: some straightforward lookups, some requiring a
     tool, a couple your corpus can't answer, and — most valuable for your
     writeup — a few where you already suspect retrieval or verification
     might struggle (ambiguous phrasing, a term that appears in multiple
     sections, a claim near a page/section boundary).

2. **Run it:**
   ```bash
   python src/eval.py --doc-id atmega32 --eval-set data/eval/atmega32_eval.jsonl
   ```
   Loads the embedding model once for the whole run (not once per
   question — Phase 4/5's per-call reload would make 20-30 questions
   painfully slow). Prints a summary and writes full per-question results
   to `data/eval/results.json`, including each raw answer, so you can
   actually read what happened on a flagged or low-precision question
   instead of just staring at an aggregate number.

3. **Test the mechanics without an API key:**
   ```bash
   python scripts/test_eval_mechanics.py
   ```
   Covers keyword matching, precision/recall arithmetic against a small
   fake corpus, the refusal heuristic, summary aggregation, and `run_eval`
   end-to-end with the LLM calls mocked. As with the other test scripts,
   this validates the harness's arithmetic is correct — not that a real
   model's answers are good.

### Design notes (Phase 6)
- **Why keyword-based ground truth instead of exact chunk IDs?** Covered
  above, but worth restating since it's the single biggest robustness
  decision in this phase: chunk IDs are an implementation detail of
  `chunk_docs.py`'s current run, not a stable identifier. An eval set tied
  to them breaks silently (wrong precision/recall, no error) the next time
  you tweak chunking. Keywords describing what a relevant chunk should be
  *about* survive that.
- **Why measure tool-use compliance as its own metric, not just fold it
  into faithfulness?** Because they catch different failure modes. A
  question can have high faithfulness (the verifier finds the answer's
  claims "supported") while still having silently skipped the calculator
  tool — the ADC prescaler case from Phase 5 testing is exactly this: the
  computed value should get flagged `unsupported` by the verifier too, but
  tool-compliance catches the skip directly and explicitly rather than
  relying entirely on the verifier catching it downstream.
- **Known limitation:** faithfulness here is "fraction of claims the
  verifier marked supported" — it inherits whatever accuracy the verifier
  itself has, including the citation-mis-attribution issue documented in
  Phase 5. A high faithfulness score means "the verifier didn't flag
  anything," which is meaningfully different from "every claim is
  definitely correctly grounded." Worth a sentence acknowledging this in
  your writeup rather than presenting faithfulness as ground truth.
- **Known limitation:** the refusal heuristic (`looks_like_refusal`) is a
  fixed list of English phrases and will miss a valid refusal phrased
  differently, or false-positive on an answer that happens to contain a
  phrase like "not mentioned" while discussing something the datasheet
  *does* cover. It's a triage signal to point you at which answers to read,
  not a metric to report on its own.


## Repo layout
```
FINDINGS.md         # the actual evidence/results writeup -- start here
data/raw/         # source PDFs (gitignored — datasheets are large)
data/processed/   # ingest.py + chunk_docs.py output (JSON/JSONL)
data/index/        # build_index.py output: FAISS index, BM25 corpus, metadata (gitignored)
src/               # pipeline code
scripts/           # one-off helpers + mechanics/unit tests
.env.example       # LLM provider config template (copy to .env)
```
