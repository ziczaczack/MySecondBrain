"""Tests for the LLM provider seam (kb.llm) and model resolution (kb.config).

No live API calls are made: we only exercise constructor-time validation
(missing API key) and the pure ``synthesis_model`` resolver. Conventions mirror
test_query.py / test_zeroconfig.py: pytest + monkeypatch, env overrides via
monkeypatch.setenv / monkeypatch.delenv.
"""

import pytest

from kb.config import synthesis_model
from kb.llm import ClaudeProvider, KbLLMError


def test_missing_api_key_raises(monkeypatch):
    """ClaudeProvider() raises KbLLMError when ANTHROPIC_API_KEY is unset."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(KbLLMError):
        ClaudeProvider()


def test_empty_api_key_raises(monkeypatch):
    """An empty ANTHROPIC_API_KEY is treated as missing -> KbLLMError."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    with pytest.raises(KbLLMError):
        ClaudeProvider()


def test_synthesis_model_default(monkeypatch, tmp_path):
    """With no KB_MODEL and no config.json, synthesis_model() is the built-in default."""
    monkeypatch.delenv("KB_MODEL", raising=False)
    # Redirect kb_home to an empty dir so no stray config.json interferes.
    monkeypatch.setenv("KB_HOME", str(tmp_path / "kbhome"))

    assert synthesis_model() == "claude-opus-4-8"


def test_synthesis_model_env_override_wins(monkeypatch, tmp_path):
    """A non-empty KB_MODEL overrides the default."""
    monkeypatch.setenv("KB_HOME", str(tmp_path / "kbhome"))
    monkeypatch.setenv("KB_MODEL", "claude-custom-model-xyz")

    assert synthesis_model() == "claude-custom-model-xyz"


def test_synthesis_model_empty_env_treated_as_unset(monkeypatch, tmp_path):
    """An empty KB_MODEL falls through to the default (treated as unset)."""
    monkeypatch.setenv("KB_HOME", str(tmp_path / "kbhome"))
    monkeypatch.setenv("KB_MODEL", "")

    assert synthesis_model() == "claude-opus-4-8"
