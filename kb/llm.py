"""LLM provider seam for the kb question-answering pipeline.

``LLMProvider`` is the protocol between the query layer and whichever language
model backend is active.  Implement the single-method protocol below to swap in
a new backend without touching ``query.py``.

Protocol quick-reference
------------------------
:class:`LLMProvider`   — backend that turns (system, context, question) → answer.
:class:`KbLLMError`    — single kb-level error type raised for all provider failures.
:class:`ClaudeProvider` — Anthropic backend: claude-opus-4-8, adaptive thinking,
                          streaming + get_final_message().

Adding a new provider (no edits to query.py required)
------------------------------------------------------
Here is how an ``OllamaProvider`` would slot in::

    class OllamaProvider:
        \"\"\"Run a local Ollama model as the answer backend.\"\"\"

        def __init__(self, model: str = "llama3") -> None:
            self._model = model

        def complete(self, system: str, context: str, question: str) -> str:
            import ollama
            resp = ollama.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"{context}\\n\\n{question}"},
                ],
            )
            return resp["message"]["content"]

    # The query layer accepts any object satisfying LLMProvider, so nothing
    # there changes:
    #
    #   answer = OllamaProvider().complete(system, context, question)
"""

from __future__ import annotations

import os

from kb.config import synthesis_model

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# kb-level error type — every provider funnels failures through this
# ---------------------------------------------------------------------------

class KbLLMError(Exception):
    """Raised when an LLM provider encounters an unrecoverable error.

    Callers can catch this single type regardless of which backend is active.
    The message always describes the root cause in human-readable terms.
    """


# ---------------------------------------------------------------------------
# LLMProvider protocol
# ---------------------------------------------------------------------------

class LLMProvider(Protocol):
    """Backend that produces a text answer from a system prompt and context.

    Implement :meth:`complete` to plug a new language model into the query
    pipeline without modifying ``query.py``.
    """

    def complete(self, system: str, context: str, question: str) -> str:
        """Return a text answer grounded in *context*.

        Parameters
        ----------
        system:
            Model instructions: persona, output format, citation style, etc.
        context:
            Retrieved passage text assembled by the query layer.  May be an
            empty string when no relevant chunks were found.
        question:
            The raw user query.
        """
        ...


# ---------------------------------------------------------------------------
# ClaudeProvider — Anthropic backend shipped today
# ---------------------------------------------------------------------------

class ClaudeProvider:
    """Anthropic backend using the configured synthesis model.

    On construction reads ``ANTHROPIC_API_KEY`` from the environment and
    raises :class:`KbLLMError` immediately if it is absent or if the
    ``anthropic`` package is not installed — so callers learn of
    misconfiguration before issuing any query.

    Every API call uses ``messages.create`` (non-streaming) with a 60-second
    client timeout, so the call can never hang indefinitely.  Only ``text``
    content blocks are returned to the caller.

    All SDK exceptions (:class:`~anthropic.APIStatusError`,
    :class:`~anthropic.APIConnectionError`, and the base
    :class:`~anthropic.APIError`) are caught and re-raised as
    :class:`KbLLMError`.
    """

    _MAX_TOKENS = 4_096

    def __init__(self) -> None:
        self._model = synthesis_model()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise KbLLMError(
                "ANTHROPIC_API_KEY is not set. "
                "Export it before running kb:\n"
                "  export ANTHROPIC_API_KEY=sk-ant-..."
            )
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise KbLLMError(
                "The 'anthropic' package is not installed. "
                "Run: pip install anthropic"
            ) from exc

        self._client = _anthropic.Anthropic(api_key=api_key, timeout=60.0)
        self._anthropic = _anthropic

    def complete(self, system: str, context: str, question: str) -> str:
        """Call the Claude API and return the answer text.

        The user turn is ``<context>\\n\\n<question>`` when context is
        non-empty, or just ``<question>`` otherwise.

        Raises :class:`KbLLMError` on any SDK-level failure, including timeouts.
        """
        user_content = f"{context}\n\n{question}" if context.strip() else question

        # Wrap the system string in a content-block list so Anthropic's prompt
        # cache can store the encoded representation after the first request.
        system_param = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._MAX_TOKENS,
                system=system_param,
                messages=[{"role": "user", "content": user_content}],
            )
        except self._anthropic.APIStatusError as exc:
            raise KbLLMError(
                f"Claude API error (HTTP {exc.status_code}): {exc.message}"
            ) from exc
        except self._anthropic.APIConnectionError as exc:
            raise KbLLMError(f"Claude connection error: {exc}") from exc
        except self._anthropic.APIError as exc:
            raise KbLLMError(f"Claude API error: {exc}") from exc

        text_parts = [
            block.text
            for block in message.content
            if block.type == "text"
        ]
        return "\n".join(text_parts)
