# Example Run

This transcript was produced by running the real pipeline (`src/graph.py`, `src/qa.py`, `src/cli.py:format_briefing`) end-to-end, with the network-touching leaves swapped for deterministic fakes (see `generate_demo.py`) because this sandbox cannot reach arxiv.org or the Gemini API. The routing, chunking, vector search, and fallback-on-parse-failure logic below is real, not mocked.

## Input
```
recent work on KV-cache compression for LLMs
```

## Pipeline log (note the PDF fetch fails and the graph degrades gracefully to abstract-only rather than crashing)
```
[log] Query understood as a topic search: 'recent work on KV-cache compression for LLMs'
[log] Retrieved 1 candidate paper(s) for topic search.
[log] Ranked 1 candidates by embedding similarity. Top pick: 2401.12345 (0.93).
[log] Selected paper: Selective KV-Cache Eviction for Memory-Efficient LLM Inference (2401.12345)
[log] Parsing degraded: PDF download failed (ConnectionError(MaxRetryError('HTTPSConnectionPool(host=\'example.invalid\', port=443): Max retries exceeded with url: /paper.pdf (Caused by NameResolutionError("HTTPSConnection(host=\'example.invalid\', port=443): Failed to resolve \'example.invalid\' ([Errno -2] Name or service not known)"))'))); using abstract only.
[log] Chunked into 1 piece(s) and embedded into the vector store.
[log] Generated executive briefing.
```

## Executive Briefing Output

# Selective KV-Cache Eviction for Memory-Efficient LLM Inference

**Authors:** J. Researcher, A. Coauthor
**arXiv ID:** 2401.12345  |  **Published:** None
**Link:** https://arxiv.org/abs/2401.12345

## Summary
This paper proposes a KV-cache compression scheme for transformer-based LLM inference that reduces memory footprint by selectively evicting low-attention-score cache entries. It matters because KV-cache size is a major bottleneck for long-context serving, and the approach reports substantial memory savings with minimal quality loss.

## Problem Statement
As context lengths grow, the key-value cache in transformer decoders grows linearly with sequence length, becoming the dominant memory cost during inference and limiting batch size / context length on fixed hardware.

## Method
- Track per-token attention scores across recent decoding steps
- Periodically evict cache entries below a dynamic importance threshold
- Use a small recovery buffer to reduce quality loss from mis-evicted tokens

## Key Results
- Up to 4.2x reduction in KV-cache memory on 7B and 13B parameter models
- Less than 0.5 perplexity degradation on long-context benchmarks
- Throughput improvement of ~1.8x at fixed batch size due to freed memory

## Limitations
- Evaluated primarily on English text; multilingual/code behavior untested
- Eviction threshold is tuned per-model and not shown to transfer automatically
- No evaluation against adversarial or retrieval-heavy long-context tasks

## Suggested Follow-up Questions
- How does the eviction threshold get chosen for a new model?
- What happens to accuracy at extreme compression ratios (>8x)?
- Was this compared against quantization-based KV-cache methods?

## Sample QA Exchanges

**Q: How is the eviction threshold chosen?**

The eviction threshold is set dynamically per layer based on a moving average of attention scores over the last N decoding steps, calibrated on a small held-out validation set for each model (Section 3.2).

*(grounded=True)*

**Q: Was this compared against quantization-based KV-cache methods?**

I couldn't find this in the paper's text — it may not be covered, or it may be in a figure/table that wasn't extracted.

*(grounded=False)*

**Q: What is the main method used?**

The method reduces KV-cache memory by evicting low-importance entries while keeping a small recovery buffer to limit quality loss (Section 3).

*(grounded=True)*
