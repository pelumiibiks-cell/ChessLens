"""Ingest public-domain instructional chess books from Project Gutenberg.
This is where the principle-level language lives - "the pin along the
e-file is a common tactical motif" reads nothing like an opening database
entry, and that's exactly the register a coaching explanation should
borrow from.
"""

from __future__ import annotations

import re

from chesslens.rag.ingest._http import fetch_text
from chesslens.rag.ingest.tagging import auto_tags
from chesslens.rag.schema import Chunk

# Gutenberg IDs confirmed by fetching the text directly - add more titles
# here once their ID is confirmed the same way; a guessed ID either 404s or
# (worse) silently pulls the wrong book, and the catalog search endpoint
# wasn't reliable enough during ingest to auto-discover them.
BOOKS = [
    {
        "id": "capablanca_fundamentals",
        "title": "Chess Fundamentals",
        "author": "Jose Raul Capablanca",
        "gutenberg_id": 33870,
    },
]

_ILLUSTRATION_RE = re.compile(r"\[Illustration\]", re.IGNORECASE)
_PAGE_MARKER_RE = re.compile(r"\{\d+\}")
_MIN_CHUNK_LEN = 80


def _gutenberg_url(gutenberg_id: int) -> str:
    return f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"


def _strip_boilerplate(raw_text: str) -> str:
    start_match = re.search(r"\*\*\* ?START OF.*?\*\*\*", raw_text)
    end_match = re.search(r"\*\*\* ?END OF.*?\*\*\*", raw_text)
    start = raw_text.index("\n", start_match.end()) + 1 if start_match else 0
    end = end_match.start() if end_match else len(raw_text)
    return raw_text[start:end]


def _looks_like_prose(paragraph: str) -> bool:
    if len(paragraph) < _MIN_CHUNK_LEN:
        return False
    letters = [c for c in paragraph if c.isalpha()]
    if not letters:
        return False
    lowercase_ratio = sum(1 for c in letters if c.islower()) / len(letters)
    return lowercase_ratio > 0.6  # filters out ALL-CAPS headers and the table of contents


def _chunk_body(body: str) -> list[str]:
    cleaned = _ILLUSTRATION_RE.sub("", body)
    cleaned = _PAGE_MARKER_RE.sub("", cleaned)
    paragraphs = re.split(r"\n\s*\n", cleaned)
    chunks = []
    for p in paragraphs:
        normalized = re.sub(r"\s+", " ", p).strip()
        if _looks_like_prose(normalized):
            chunks.append(normalized)
    return chunks


def fetch_book_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for book in BOOKS:
        try:
            raw_text = fetch_text(_gutenberg_url(book["gutenberg_id"]))
        except Exception:
            continue  # a missing/renumbered Gutenberg id shouldn't fail the whole build
        body = _strip_boilerplate(raw_text)
        url = f"https://www.gutenberg.org/ebooks/{book['gutenberg_id']}"
        for i, paragraph in enumerate(_chunk_body(body)):
            chunks.append(
                Chunk(
                    id=f"books:{book['id']}:{i:04d}",
                    source="books",
                    title=f"{book['title']} - {book['author']}",
                    text=paragraph,
                    eco=None,
                    epd=None,
                    tags=auto_tags(paragraph),
                    url=url,
                )
            )
    return chunks
