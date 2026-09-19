# Autonomous arXiv Paper Digest & QA Agent

An agent that takes a research topic or a specific arXiv paper ID, retrieves
the paper, produces a structured executive briefing, and answers grounded
follow-up questions about it (RAG-based QA).

## Quickstart

```bash
git clone <this-repo-url>
cd arxiv_digest_agent
pip install -r requirements.txt
cp .env.example .env
# edit .env: paste a free Gemini key into GEMINI_API_KEY (get one at https://aistudio.google.com/apikey)

python -m src.cli "recent work on KV-cache compression for LLMs"
```

`query` is a required argument — running `python -m src.cli` with nothing
after it will show a usage error with examples, same as any CLI tool
(`git commit` with no message behaves the same way). Run `python -m src.cli -h`
for full usage.

No API key handy yet? Run the test suite instead — it needs no key and no
internet, and proves the core logic works:
```bash
pip install pytest && python -m pytest tests/ -v
```

### Troubleshooting: "model not found" (404)

Google's free-tier model lineup changes over time; if `GEMINI_MODEL` in your
`.env` 404s, list what your key actually supports and pick a current name:
```bash
python -c "from google import genai; import os; from dotenv import load_dotenv; load_dotenv(); c = genai.Client(api_key=os.environ['GEMINI_API_KEY']); [print(m.name) for m in c.models.list()]"
```

Built as a plain Python state graph (no LangGraph/CrewAI) with local,
free-tier-friendly components: Gemini free tier for the LLM, a local
`sentence-transformers` model for embeddings, and FAISS for vector search —
no cloud vector DB, no paid API keys required.

---

## 1. Architecture: the state graph

```
query_understanding
        │
        ▼
     retrieve ──(topic, zero results)──► END (NEEDS_INPUT)
        │                │
        │ (paper_id)     │ (topic, N≥1 results)
        ▼                ▼
  (found?) ──(no)──► END (ERROR)   rank_select
        │                          │
        │ (yes)                   │ (always)
        ▼                         ▼
     fetch_parse ◄─────────────────┘
        │  (parse failure or scanned/huge PDF → degrade, don't crash)
        ▼
     chunk_embed
        │
        ▼
     summarize
        │
        ▼
       END (status = OK | DEGRADED)
                │
                ▼ (interactive, outside the one-shot graph)
             QA loop  (repeatable: question → retrieve top-k chunks → grounded answer)
```

**State shape** (`src/state.py: AgentState`) — the single object every node
reads from and writes to:

| Field | Set by | Used by |
|---|---|---|
| `mode` (topic / paper_id) | `query_understanding` | `retrieve` |
| `candidates: list[PaperMetadata]` | `retrieve` | `rank_select` |
| `selected_paper: PaperMetadata` | `retrieve` or `rank_select` | `fetch_parse`, `summarize`, `qa` |
| `parsed: ParsedPaper` (+ `parse_degraded` flag) | `fetch_parse` | `chunk_embed`, `summarize` |
| `chunks`, `vector_collection` | `chunk_embed` | QA loop |
| `briefing: dict` | `summarize` | CLI output |
| `qa_history: list[QATurn]` | QA loop | CLI output / future multi-turn context |
| `status`, `messages` | every node | CLI output, tests |

**Why a hand-rolled state machine instead of LangGraph/CrewAI:** the
pipeline is mostly linear with a small number of conditional branches (zero
results, paper-not-found, parse degradation). A framework's value is in
managing many/cyclical branches, checkpointing, and multi-agent handoffs —
none of which this pipeline needs. Each node here is `(AgentState, **deps)
-> (AgentState, next_node_name | None)`, which is trivially testable with
fakes (see `tests/test_graph.py`) without spinning up a framework runtime.
This was the single biggest design tradeoff in the project — documented
further in §5.

**Why the QA loop is not a graph node:** it's interactive and repeated
(0..N questions), whereas every other stage runs exactly once per paper.
Folding it into the same graph would mean either a self-loop edge (extra
complexity for no real benefit here) or re-running the whole graph per
question (wasteful — the vector store and briefing are already built). It
reads the same `AgentState` the graph produced, so state continuity is
still explicit, just via a plain function (`src/qa.py: ask`) called
per-question instead of a graph transition.

---

## 2. Project layout

```
src/
  state.py          # AgentState, PaperMetadata, ParsedPaper, QATurn (the shared state)
  arxiv_client.py    # arXiv API: topic search + direct ID fetch, ID/URL detection
  ranking.py         # embedding-similarity re-ranking of topic-search candidates
  pdf_parser.py       # PDF fetch + text extraction, with 3 documented failure fallbacks
  chunker.py          # word-count chunking with overlap
  embeddings.py        # sentence-transformers wrapper (local, no API key)
  vector_store.py       # FAISS-backed per-paper vector store + registry
  llm.py                 # LLMClient interface + GeminiClient implementation
  graph.py                # the state graph: nodes, edges, routing, run_graph()
  qa.py                     # RAG QA loop, grounded-or-refuse
  cli.py                     # entry point, briefing formatting, interactive QA
tests/
  test_chunker.py     # chunking correctness (no word loss, overlap behavior)
  test_graph.py        # routing/branching logic with fake LLM/embedder/arXiv (offline, fast)
examples/
  generate_demo.py      # produces examples/sample_run.md (see §4)
  sample_run.md           # a full example transcript
```

---

## 3. Setup & run

Requires Python 3.10+.

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and set GEMINI_API_KEY (free key: https://aistudio.google.com/apikey)
# LLM_PROVIDER=gemini is the default; see "Choosing an LLM backend" below.
```

### Choosing an LLM backend

`src/llm.py` defines `LLMClient` as an abstract interface with two real
implementations: `GeminiClient` (default) and `GroqClient` (alternative).
Switch by setting `LLM_PROVIDER=gemini` or `LLM_PROVIDER=groq` in `.env` —
no code changes needed.

**Gemini is the default and recommended choice for this specific task**
because of its very large free-tier context window (~1M tokens on
`gemini-1.5-flash`), which matters here: full papers can run 15,000-30,000+
words, and a small context window forces aggressive truncation or a more
complex map-reduce summarization step just to fit. Gemini also has solid
native JSON mode, which the structured-briefing output relies on.

Groq is included as a genuine second option, not just a mention: it's
much faster and has a very generous free rate limit, at the cost of a
smaller context window (8K-32K depending on model, vs. Gemini's ~1M) — so
paper text is trimmed harder before summarization (see `GroqClient.TRIMMED_CHARS`
in `src/llm.py`). Worth switching to if you hit Gemini's ~15 req/min limit
during testing, or want faster iteration on shorter papers.

```bash
# Topic search
python -m src.cli "recent work on KV-cache compression for LLMs"

# Specific paper
python -m src.cli 2401.12345
python -m src.cli https://arxiv.org/abs/2401.12345

# Write briefing to disk and skip the interactive QA loop (useful for scripting)
python -m src.cli "..." --json out/briefing.json --no-interactive
```

First run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) from
Hugging Face — this needs one-time internet access even though inference
after that is fully local.

**Free-tier rate limits:** `gemini-1.5-flash`'s no-cost tier is roughly 15
requests/minute. This pipeline makes 1 call for the briefing plus 1 per QA
question — comfortably under the limit for normal use; batching many
papers back-to-back in a script could hit it.

Run the test suite (no API key or internet needed — everything is faked):

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## 4. Example run

See [`examples/sample_run.md`](examples/sample_run.md) for a full transcript
(input → pipeline log → briefing → 3 QA exchanges).

**Important note on how it was produced:** this sandbox's network access is
locked to package registries only (no route to `arxiv.org`, `generativelanguage.googleapis.com`,
or `huggingface.co`), so I could not execute a live end-to-end run inside
this environment. `examples/generate_demo.py` runs the *real* graph and QA
code (`src/graph.py`, `src/qa.py`, `src/cli.py:format_briefing` — nothing
mocked there) but swaps the three network-touching leaves
(`arxiv_client.search_by_topic`, the LLM, the embedder) for deterministic
fakes with the same interface as the real `GeminiClient`/`EmbeddingModel`.
The transcript therefore proves the routing, chunking, vector search, and
graceful-degradation logic genuinely work — it does not prove Gemini
produces a good summary for a real paper, which requires a key and network
access I don't have here.

What **was** verified directly, without any mocking, in this sandbox:
- All 9 automated tests pass (`pytest tests/ -v`), covering: zero-result
  topic search, paper-ID-not-found, the happy path through to a briefing,
  custom candidate-selection callbacks, and QA grounding-refusal behavior.
- The ranking algorithm and FAISS vector store were verified against a
  hand-built mock embedder with known correct answers (retrieves the right
  chunk / ranks the right paper first) — see the commit history / can be
  re-run via the snippets in `tests/`.
- Every module imports cleanly with the real dependencies installed
  (`sentence-transformers`, `faiss-cpu`, `PyMuPDF`, `google-generativeai`).
- The PDF-parser fallback path was exercised against a real (unreachable)
  URL and correctly degraded to abstract-only rather than raising.

What genuinely needs a real run to confirm: Gemini's JSON-mode reliability
on an actual paper's full text, and how well `all-MiniLM-L6-v2` embeddings
retrieve the *right* chunk on a real 15+ page paper (vs. the toy 3-chunk
example in the mocked tests).

---

## 5. Design decisions & tradeoffs

**A note on API churn:** during development, Google retired the entire
Gemini 2.x free-tier line (`gemini-1.5-flash`, `gemini-2.0-flash`) and moved
to a Gemini 3.x line (current default here: `gemini-3.6-flash`), and
deprecated the `temperature`/`top_p`/`top_k` sampling parameters in the
process. If `GEMINI_MODEL` in your `.env` 404s by the time you run this,
Google has likely shipped another model update — see
`https://ai.google.dev/gemini-api/docs/models` for the current lineup, or
run the model-listing snippet in the troubleshooting section below.

**Ranking: embedding similarity, not a second LLM call.** Re-ranking
candidates by cosine similarity between the query and each abstract (same
embedding space as chunk retrieval) is cheap and dependency-light. An LLM
re-ranker would likely handle nuanced queries better ("papers that *extend*
X, not just cite it") but costs an extra API call and adds another
JSON-parsing failure surface for marginal benefit at this scale (≤8
candidates).

**Candidate selection: auto-pick the top-ranked paper, not an interactive
prompt.** `rank_select` logs the top-3 candidates with scores either way,
but defaults to auto-selecting #1 so the CLI runs non-interactively (good
for scripting/testing). `run_graph()` accepts a `select_callback` for
exactly this reason — a future UI or a `--interactive-select` CLI flag
could plug in a picker without touching the graph itself. With more time
I'd wire that flag up.

**Chunking: fixed word-count windows with overlap, not semantic
chunking.** Predictable and robust across papers with wildly inconsistent
PDF-to-text layouts (two-column layouts, inline equations, etc.), at the
cost of occasionally splitting a paragraph mid-thought. A layout-aware or
sentence-boundary-aware chunker would produce cleaner chunks; I'd try
`sections` (already extracted by the regex-based section splitter) as
chunk boundaries first, before reaching for anything heavier.

**Section splitting: regex over common headings, not a layout model.**
Good enough to hand the summarizer some structure; not trustworthy for
citation-grade extraction. Papers with unconventional heading text (or none
at all, e.g. some older/differently-formatted papers) fall through to
`sections = {}` and the summarizer just works off the full text, which is
still correct, just less structured.

**Grounding: a literal refusal token, not a confidence score.** The QA
prompt instructs the model to output exactly `NOT_FOUND_IN_PAPER` when the
retrieved chunks don't answer the question, and the code checks for that
token rather than trying to infer confidence from prose. Blunt, but
reliable — a model second-guessing itself into a hedge-y non-answer is
worse than an unambiguous "not found," and this way `QATurn.grounded` is a
clean boolean the UI/tests can rely on instead of parsing tone.

**Failure handling (§5 in the assessment):**
- *Zero/many arXiv results:* zero → `NEEDS_INPUT` status with a message
  asking the user to broaden the query, no crash. Many → ranked, top-1
  auto-selected, runners-up logged (see selection tradeoff above).
- *PDF fails to parse (scanned/broken/huge):* `pdf_parser.py` has three
  independent fallback points (download failure, PyMuPDF open failure,
  low text-density heuristic for scanned PDFs) that all converge on the
  same behavior — fall back to the arXiv abstract, flag
  `parse_degraded=True`, and let the rest of the pipeline run on the
  abstract alone. Huge papers are capped at `MAX_PAGES=60` rather than
  failing outright. The briefing prompt is told explicitly when it's
  working from an abstract only, so it doesn't confidently invent
  full-paper detail.
- *QA grounding:* see above.
- *State persistence between summarize and QA:* in-process, in-memory for
  this CLI scope (`VectorStoreRegistry` keyed by `arxiv_id`, held for the
  process lifetime). `PaperVectorStore.save()/load()` already exist for
  disk persistence — the natural next step for a multi-session or web
  version would be a session object (e.g. keyed by a session ID) that
  persists `AgentState` + calls `save()`/`load()` instead of holding
  everything in one process's memory.

**What I'd do differently with more time:**
1. Wire up interactive candidate selection in the CLI (`select_callback` is
   already graph-ready, just not exposed as a flag).
2. Replace regex section-splitting with sentence-boundary-aware chunking
   that respects the extracted `sections` as hard boundaries.
3. Add a lightweight LLM-based re-ranker as an opt-in alternative to the
   embedding-similarity ranker for topic queries where recall from arXiv's
   own search is poor.
4. Persist `AgentState`+vector store to disk by default (not just via the
   unused `save/load` methods) so a QA session survives a process restart.
5. A proper eval set (a handful of papers with hand-written Q/A pairs) to
   measure retrieval precision instead of relying on spot-checks.

---

## 6. Constraints checklist

- [x] No paid API keys required (Gemini free tier; document rate limits above)
- [x] Local vector DB (FAISS, in-process)
- [x] Official arXiv API only, no scraping
- [x] Python
- [x] CLI interaction only, no frontend
- [x] No fine-tuning, no auth/deployment infra, arXiv-only sources
