"""The corpus chunk shape. Every retrievable piece of text - an opening
line, a Wikibooks section, a paragraph from a public-domain instructional
book - is normalized to this before it goes in the store, so retrieval and
citation don't need to know which ingest source a chunk came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str  # stable, e.g. "openings:B12:0007", "wikibooks:Chess_Opening_Theory/1._e4:3"
    source: str  # "openings" | "wikibooks" | "books"
    title: str  # opening name / page title / book section heading
    text: str  # the retrievable content
    eco: str | None = None  # ECO code, when known (openings, some wikibooks pages)
    epd: str | None = None  # exact position key (board.epd()) - openings only, exact-match channel
    tags: list[str] = field(default_factory=list)  # motif/structure vocabulary found in the text
    url: str | None = None  # citation source, when the ingest source has a stable URL

    # Additional positions this chunk is relevant to besides `epd` itself -
    # e.g. every intermediate position along an opening line's move
    # sequence, not just the final one. lichess-org/chess-openings only
    # names branch points, so the position one ply past the last named node
    # (still completely standard theory) has no exact EPD of its own unless
    # something indexes it here.
    extra_epds: list[str] = field(default_factory=list)
