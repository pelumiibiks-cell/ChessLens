"""Ingest the lichess-org/chess-openings TSVs (~3500 named lines, ECO A-E)
into corpus chunks. Each line is replayed on a board so the chunk carries
the exact EPD of the position it leads to - that's what makes the EPD
retrieval channel exact rather than fuzzy, and it also picks up whatever
structural archetype (IQP, Carlsbad, ...) the line resolves into.
"""

from __future__ import annotations

import re

import chess

from chesslens.features.structures import classify_pawn_structure
from chesslens.rag.ingest._http import fetch_text
from chesslens.rag.schema import Chunk

BASE_URL = "https://raw.githubusercontent.com/lichess-org/chess-openings/master"
FILES = ["a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv"]
REPO_URL = "https://github.com/lichess-org/chess-openings"

_MOVE_NUMBER_RE = re.compile(r"\d+\.(\.\.)?\s*")


def _replay(pgn_movetext: str) -> tuple[chess.Board, list[str]] | tuple[None, None]:
    """Returns the final board and the EPD after every move along the way -
    the intermediate positions matter because lichess-org/chess-openings
    only names branch points, so the position one ply past the last named
    node (still completely standard theory) would otherwise have no exact
    EPD anywhere in the corpus.
    """
    board = chess.Board()
    intermediate_epds = []
    cleaned = _MOVE_NUMBER_RE.sub("", pgn_movetext).strip()
    for token in cleaned.split():
        try:
            board.push_san(token)
        except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
            return None, None
        intermediate_epds.append(board.epd())
    return board, intermediate_epds


def _parse_tsv(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split("\t")
        rows.append(dict(zip(header, values)))
    return rows


def fetch_opening_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for filename in FILES:
        text = fetch_text(f"{BASE_URL}/{filename}")
        rows = _parse_tsv(text)
        for i, row in enumerate(rows):
            eco, name, pgn = row.get("eco", ""), row.get("name", ""), row.get("pgn", "")
            board, intermediate_epds = _replay(pgn)
            tags = [s.name for s in classify_pawn_structure(board)] if board is not None else []
            chunks.append(
                Chunk(
                    id=f"openings:{eco}:{i:04d}",
                    source="openings",
                    title=name,
                    text=f"{name} ({eco}): {pgn}",
                    eco=eco or None,
                    epd=board.epd() if board is not None else None,
                    tags=tags,
                    url=REPO_URL,
                    extra_epds=intermediate_epds or [],
                )
            )
    return chunks
