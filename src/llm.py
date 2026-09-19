"""
LLM client wrapper. Abstracted behind `LLMClient` so the orchestration code
(graph.py) never talks to a vendor SDK directly — swapping Gemini for a
local Ollama model or another free-tier API only means writing a new class
here, not touching the graph.

Two implementations are provided:
    - `GeminiClient` (default): uses the current `google-genai` SDK (the
      `google-generativeai` package is deprecated and no longer receives
      updates). Best fit for this task because of its huge free-tier
      context window (~1M tokens), which matters when feeding full paper
      text to the summarizer. Native JSON mode via response_mime_type.
      Default model is gemini-3.6-flash (Google's Gemini 2.x line, including
      2.0/2.5-flash, has been retired as of this writing; the `temperature`
      sampling parameter is also deprecated on the 3.x model line and is
      intentionally not set here - see Google's release notes if this
      changes again by the time you read this).
    - `GroqClient` (alternative): much smaller context window (8K-32K
      depending on model) but very fast and a generous free rate limit.
      Useful as a fallback if you hit Gemini's ~15 req/min limit, or if you
      want faster iteration during development. Uses the OpenAI-compatible
      chat completions API that Groq exposes.

Select via the LLM_PROVIDER env var ("gemini" or "groq"); see get_llm_client()
at the bottom of this file. Gemini is used if LLM_PROVIDER is unset.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

NOT_FOUND_MARKER = "NOT_FOUND_IN_PAPER"

_SUMMARY_SYSTEM_PROMPT = """You are a research assistant producing an executive briefing for a busy \
engineer deciding whether a paper is worth a deep read. Be precise and concrete. \
Do not invent results, numbers, or claims that are not supported by the provided text. \
If the provided text is only an abstract (not the full paper), work only from what is given \
and do not fabricate details about sections you cannot see.

Return ONLY a JSON object with exactly these keys:
{
  "summary": "1 paragraph, plain English, explaining why this paper matters",
  "problem_statement": "1-3 sentences",
  "method": ["bullet point", "bullet point", "..."],
  "key_results": ["bullet point", "bullet point", "..."],
  "limitations": ["bullet point", "bullet point", "..."],
  "suggested_questions": ["question a reader might ask", "..."]
}
Do not include markdown code fences. Do not include any text outside the JSON object.
If limitations are not explicitly discussed in the text, infer reasonable ones from the \
paper's own scope/claims rather than leaving the list empty, and say they are inferred.
"""

_QA_SYSTEM_PROMPT = f"""You answer questions about a specific paper using ONLY the excerpts \
provided below. Do not use outside knowledge and do not guess. If the excerpts do not contain \
the answer, respond with exactly this token and nothing else: {NOT_FOUND_MARKER}

Keep answers concise (2-5 sentences) and, where useful, note which part of the excerpt supports \
the claim.
"""


class LLMClient(ABC):
    @abstractmethod
    def summarize(self, paper_title: str, text: str, degraded: bool) -> dict:
        ...

    @abstractmethod
    def answer(self, question: str, context_chunks: list[str], paper_title: str) -> str:
        ...


class GeminiClient(LLMClient):
    def __init__(self, model_name: str | None = None):
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and put it in your .env file."
            )
        self.model_name = model_name or os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
        self._client = genai.Client(api_key=api_key)
        self._types = types

    def summarize(self, paper_title: str, text: str, degraded: bool) -> dict:
        degraded_note = (
            "\n\nNOTE: only the abstract was available for this paper (full-text parsing failed "
            "or was unavailable). Base the briefing on the abstract only and reflect that "
            "limited depth honestly in the summary."
            if degraded
            else ""
        )
        # keep the prompt within a sane token budget for the free tier
        trimmed_text = text[:24000]

        prompt = (
            f"{_SUMMARY_SYSTEM_PROMPT}{degraded_note}\n\n"
            f"PAPER TITLE: {paper_title}\n\nTEXT:\n{trimmed_text}"
        )
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: model didn't respect JSON mode (rare, but happens on some free-tier
            # responses). Wrap the raw text so the pipeline doesn't crash downstream.
            return {
                "summary": raw,
                "problem_statement": "",
                "method": [],
                "key_results": [],
                "limitations": ["Model did not return structured JSON; raw text shown in summary."],
                "suggested_questions": [],
            }

    def answer(self, question: str, context_chunks: list[str], paper_title: str) -> str:
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no relevant excerpts found)"
        prompt = (
            f"{_QA_SYSTEM_PROMPT}\n\nPAPER TITLE: {paper_title}\n\n"
            f"EXCERPTS:\n{context}\n\nQUESTION: {question}"
        )
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text.strip()


class GroqClient(LLMClient):
    """
    Alternative LLM backend using Groq's free tier (OpenAI-compatible API).
    Much smaller context window than Gemini, so full paper text is trimmed
    more aggressively here (see TRIMMED_CHARS below) - fine for shorter
    papers, lossier for very long ones. Included to prove LLMClient is a
    real, swappable abstraction, and as a fast fallback if Gemini's rate
    limit is a problem.
    """

    # Groq's free-tier context windows vary by model; keep this conservative
    # so summarization prompts + a paper chunk reliably fit.
    TRIMMED_CHARS = 12000

    def __init__(self, model_name: str | None = None):
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys and put it in your .env file."
            )
        self.model_name = model_name or os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
        self._client = Groq(api_key=api_key)

    def summarize(self, paper_title: str, text: str, degraded: bool) -> dict:
        degraded_note = (
            "\n\nNOTE: only the abstract was available for this paper (full-text parsing failed "
            "or was unavailable). Base the briefing on the abstract only and reflect that "
            "limited depth honestly in the summary."
            if degraded
            else ""
        )
        trimmed_text = text[: self.TRIMMED_CHARS]
        prompt = (
            f"{_SUMMARY_SYSTEM_PROMPT}{degraded_note}\n\n"
            f"PAPER TITLE: {paper_title}\n\nTEXT:\n{trimmed_text}"
        )
        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "summary": raw,
                "problem_statement": "",
                "method": [],
                "key_results": [],
                "limitations": ["Model did not return structured JSON; raw text shown in summary."],
                "suggested_questions": [],
            }

    def answer(self, question: str, context_chunks: list[str], paper_title: str) -> str:
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no relevant excerpts found)"
        prompt = (
            f"{_QA_SYSTEM_PROMPT}\n\nPAPER TITLE: {paper_title}\n\n"
            f"EXCERPTS:\n{context}\n\nQUESTION: {question}"
        )
        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return completion.choices[0].message.content.strip()


def get_llm_client() -> LLMClient:
    """Factory: picks the backend from LLM_PROVIDER env var. Defaults to Gemini."""
    provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
    if provider == "groq":
        return GroqClient()
    if provider == "gemini":
        return GeminiClient()
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'; expected 'gemini' or 'groq'.")
