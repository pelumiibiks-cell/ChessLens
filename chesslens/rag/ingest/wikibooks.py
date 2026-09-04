"""Ingest the Wikibooks Chess book (en.wikibooks.org/wiki/Chess) - rules,
strategy, endgames, and opening theory in plain prose, which is exactly the
principle-level language a coach needs and an opening database doesn't
have.

The plan called for a `Category:Chess_Opening_Theory` category, but that
category doesn't exist on Wikibooks - the book's content instead lives as
subpages under the `Chess/` title prefix, so this lists those directly via
`list=allpages` (filtering out the many opening-name redirects) and pulls
plain-text extracts one page at a time instead of parsing wikitext by hand.

One page at a time, not batched: MediaWiki's `prop=extracts` silently caps
`exlimit` to 1 whenever `explaintext` requests a whole-article extract
(only the intro-only mode supports multi-page batches), so a batched
request here would return real text for one title per batch and silently
empty extracts for the rest - a bug this module hit and had to work around.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse

from chesslens.rag.ingest._http import fetch_text
from chesslens.rag.ingest.tagging import auto_tags
from chesslens.rag.schema import Chunk

API = "https://en.wikibooks.org/w/api.php"
BOOK_PREFIX = "Chess/"
REQUEST_DELAY_S = 0.5


def _list_pages() -> list[str]:
    titles: list[str] = []
    ap_continue = None
    while True:
        url = (
            f"{API}?action=query&list=allpages&apprefix={BOOK_PREFIX}"
            "&apfilterredir=nonredirects&aplimit=500&format=json"
        )
        if ap_continue:
            url += f"&apcontinue={urllib.parse.quote(ap_continue)}"
        data = json.loads(fetch_text(url))
        titles += [p["title"] for p in data["query"]["allpages"]]
        cont = data.get("continue")
        if not cont:
            break
        ap_continue = cont["apcontinue"]
    return titles


def _fetch_extract(title: str) -> str:
    encoded = urllib.parse.quote(title, safe="")
    url = f"{API}?action=query&prop=extracts&explaintext=1&titles={encoded}&format=json"
    data = json.loads(fetch_text(url))
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "")
    return ""


def _chunk_text(text: str) -> list[str]:
    paragraphs = re.split(r"\n{2,}", text)
    return [p.strip() for p in paragraphs if len(p.strip()) >= 60]


def fetch_wikibooks_chunks() -> list[Chunk]:
    titles = _list_pages()
    chunks: list[Chunk] = []
    for i, title in enumerate(titles):
        if i > 0:
            time.sleep(REQUEST_DELAY_S)  # be polite - one page per request already means ~110 calls
        extract = _fetch_extract(title)
        if not extract:
            continue
        page_slug = title.replace(" ", "_")
        for j, paragraph in enumerate(_chunk_text(extract)):
            chunks.append(
                Chunk(
                    id=f"wikibooks:{page_slug}:{j:03d}",
                    source="wikibooks",
                    title=title,
                    text=paragraph,
                    eco=None,
                    epd=None,
                    tags=auto_tags(paragraph),
                    url=f"https://en.wikibooks.org/wiki/{page_slug}",
                )
            )
    return chunks
