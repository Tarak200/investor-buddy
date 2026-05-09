"""
evaluation/prompt_versioner.py
--------------------------------
SHA-256 version tracking for all prompt files under config/prompts/v1/.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent.parent / "config" / "prompts" / "v1"


def get_version(prompt_name: str) -> str:
    """Return the SHA-256 hash of a prompt file (hex digest)."""
    path = _PROMPTS_DIR / f"{prompt_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def get_all_versions() -> dict[str, str]:
    """Return {prompt_name: sha256_hex} for all prompts in v1/."""
    versions: dict[str, str] = {}
    for prompt_file in sorted(_PROMPTS_DIR.glob("*.txt")):
        name = prompt_file.stem
        content = prompt_file.read_bytes()
        versions[name] = hashlib.sha256(content).hexdigest()
    return versions


def assert_prompt_unchanged(prompt_name: str, expected_hash: str) -> bool:
    """
    Return True if the prompt still matches the expected hash.
    Used in CI to detect accidental prompt changes.
    """
    current = get_version(prompt_name)
    return current == expected_hash
