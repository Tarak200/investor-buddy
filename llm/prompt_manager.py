"""
llm/prompt_manager.py
---------------------
Loads versioned prompts from config/prompts/v1/ by name.
Each prompt file is plain text with {placeholder} variables.
SHA256 hash of the template content is used as the version string.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "config" / "prompts" / "v1"


def load_prompt(name: str) -> str:
    """
    Load a prompt template by filename stem (without .txt extension).

    Example
    -------
    template = load_prompt("verification_judge")
    filled = template.format(raw_snippet="...", claim_value="...")
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt '{name}' not found at {path}. "
            "Create the file under config/prompts/v1/{name}.txt"
        )
    content = path.read_text(encoding="utf-8")
    version = _sha256(content)
    log.debug("prompt_loaded", name=name, version=version[:8])
    return content


def get_prompt_version(name: str) -> str:
    """Return the SHA256 version string for a prompt by name."""
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        return "unknown"
    return _sha256(path.read_text(encoding="utf-8"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
