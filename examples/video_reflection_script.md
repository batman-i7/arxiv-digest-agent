# Video Reflection Script (~4 minutes)

Read this over once, don't memorize word-for-word — speak it naturally,
pause where marked. Have the code, README, and a terminal running the CLI
open and ready to switch to when noted.

---

**[0:00–0:30] — What I built**

"Hi, I'm Likhidan. For this assessment I built an agent that takes either
a research topic or a specific arXiv paper ID, retrieves the paper,
produces a structured executive briefing, and then answers grounded
follow-up questions about it using retrieval-augmented generation.

I built this as an explicit state graph — not a single prompt chain — with
seven stages: query understanding, arXiv retrieval, ranking and selection,
PDF fetch and parse, chunking and embedding, summarization, and finally an
interactive QA loop."

*(Optional: switch to terminal, run a quick live example here if time allows — `python -m src.cli "your topic"` — otherwise continue narrating over the README architecture diagram.)*

---

**[0:30–1:15] — Architecture and state design**

"The core of the design is a single shared state object — `AgentState` —
that every node reads from and writes to. It carries things like the
detected query mode, the ranked candidate papers, the selected paper,
parsed text, chunks, the vector store reference, the generated briefing,
and the QA conversation history.

I deliberately didn't use a framework like LangGraph or CrewAI for this.
The pipeline is mostly linear with a handful of conditional branches — zero
search results, a paper ID that doesn't resolve, a PDF that fails to parse
— and a hand-rolled state machine where each node is just a function that
takes the state and returns the next node name kept things transparent and
easy to unit test with fakes, which I actually did — I have nine tests
that exercise the routing logic offline, without hitting the real arXiv
API or Gemini."

---

**[1:15–2:15] — Grounding and failure handling**

"Two things I want to call out specifically because the assessment asked
us to think about them.

First, grounding in the QA loop. Every answer has to come from retrieved
chunks in the paper — never from the model's own knowledge. I enforce this
with a literal refusal token: if the retrieved context doesn't contain the
answer, the model is instructed to output an exact marker string, and my
code checks for that token rather than trying to guess from the tone of the
response whether it's hedging. That gives me a clean boolean — grounded or
not — that the rest of the system can rely on.

Second, PDF parsing failure. This is probably the most fragile part of any
pipeline like this — scanned PDFs, broken layouts, huge papers. I built
three fallback points into the parser: if the download fails, if PyMuPDF
can't open the file, or if the extracted text density is too low to trust,
meaning it's probably a scanned image. All three converge on the same
behavior: fall back to the arXiv abstract, flag the state as degraded, and
let the rest of the pipeline continue rather than crashing the whole run.
The summarizer is explicitly told when it's only working from an abstract,
so it doesn't confidently invent details it doesn't have."

---

**[2:15–3:00] — A real design tradeoff I made**

"One tradeoff I want to be upfront about: for ranking candidate papers in
a topic search, I use embedding similarity between the query and each
abstract, rather than a second LLM call to re-rank them. It's cheaper and
has one less place for things to go wrong — no extra JSON parsing, no extra
API call — but an LLM re-ranker would probably handle more nuanced queries
better, like 'papers that extend X rather than just cite it.' At this
scale, with at most eight candidates, I judged the simpler approach was the
right call, but it's a real limitation I'd revisit if this needed to
scale to more sophisticated topic queries.

I also want to be honest that I built and tested this in an environment
without live internet access to arXiv or the Gemini API, so what I could
verify directly was the routing and branching logic using fake
implementations of the network-dependent pieces, plus the ranking and
vector-search math against a hand-built mock. All nine tests pass. What I
haven't been able to verify end-to-end myself is Gemini's actual summary
quality on a real paper — that's the one piece I'd want a first live run to
confirm."

---

**[3:00–3:40] — What I'd do differently with more time**

"With more time, there are a few things I'd prioritize. I'd wire up
interactive candidate selection — right now it auto-picks the top-ranked
paper, but the code already accepts a selection callback, so exposing that
as a CLI flag is a small change. I'd replace my regex-based section
splitting with something that respects sentence boundaries so chunks don't
get cut mid-thought. And I'd persist the vector store to disk by default
rather than keeping it in memory for the process lifetime, so a QA session
could survive a restart — the save and load methods are already there, just
not wired into the default flow."

---

**[3:40–4:00] — Close**

"Overall, I focused on making sure the pipeline degrades gracefully instead
of breaking, that QA answers are actually grounded rather than just
plausible-sounding, and that the whole thing is testable without needing
live API access every time. Thanks for reviewing — happy to walk through
any part of the code in more detail."

---

## Notes for recording

- This reads at a natural pace in about 4:00–4:15. If you're running long,
  cut the "what I'd do differently" section down to 2 bullets instead of 3.
- If you do a live terminal demo, keep it under 20 seconds — show the
  command, let the briefing print, move on. Don't wait through the full QA
  loop on camera.
- It's fine — good, even — to sound like you're explaining your own
  decisions, not reciting a script. Deviate wherever it feels more natural.
