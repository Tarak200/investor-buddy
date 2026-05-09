"""
retrieval/document_processor.py
---------------------------------
Converts PDF and HTML documents into text chunks suitable for embedding
and upserting into the Weaviate vector store.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import requests
import structlog
from bs4 import BeautifulSoup

log = structlog.get_logger(__name__)

_CHUNK_SIZE = 512       # characters per chunk
_CHUNK_OVERLAP = 64     # overlap between consecutive chunks


def chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 20]


def pdf_to_chunks(pdf_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract text from a PDF file and split into chunks.

    Returns
    -------
    list of {"page": int, "text": str}
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        chunks: list[dict[str, Any]] = []
        for page_num, page in enumerate(doc, start=1):
            raw = page.get_text("text")
            for chunk in chunk_text(raw):
                chunks.append({"page": page_num, "text": chunk})
        doc.close()
        log.info("pdf_processed", path=str(pdf_path), chunks=len(chunks))
        return chunks
    except ImportError:
        log.error("pymupdf_not_installed", path=str(pdf_path))
        return []
    except Exception as exc:
        log.error("pdf_parse_error", path=str(pdf_path), error=str(exc))
        return []


def html_to_chunks(html: str, source_url: str = "") -> list[dict[str, Any]]:
    """
    Parse HTML string, extract visible text, and split into chunks.

    Returns
    -------
    list of {"source_url": str, "text": str}
    """
    try:
        soup = BeautifulSoup(html, "lxml")
        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        raw = soup.get_text(separator=" ", strip=True)
        return [{"source_url": source_url, "text": c} for c in chunk_text(raw)]
    except Exception as exc:
        log.error("html_parse_error", url=source_url, error=str(exc))
        return []


def url_to_chunks(url: str, timeout: int = 15) -> list[dict[str, Any]]:
    """Fetch a URL and parse its HTML content into chunks."""
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; FinancialAssistant/0.1)"},
        )
        resp.raise_for_status()
        return html_to_chunks(resp.text, source_url=url)
    except Exception as exc:
        log.warning("url_fetch_error", url=url, error=str(exc))
        return []
