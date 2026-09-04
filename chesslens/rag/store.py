"""SQLite-backed corpus store: exact EPD lookup, ECO-prefix / tag structural
filtering, and BM25 full-text search - all from the standard library
(sqlite3's FTS5 extension), no vector DB or embedding model required.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable

from chesslens.rag.schema import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    eco TEXT,
    epd TEXT,
    tags TEXT NOT NULL,
    url TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_eco ON chunks(eco);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    id UNINDEXED,
    title,
    text,
    tokenize = 'porter unicode61'
);

-- Every position a chunk is relevant to, not just its own `epd` - includes
-- every intermediate position along an opening line's move sequence (see
-- Chunk.extra_epds), so "is this EPD known theory" and exact-position
-- retrieval both work one ply past the last explicitly named branch point.
CREATE TABLE IF NOT EXISTS epd_index (
    epd TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (epd, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_epd_index_epd ON epd_index(epd);
"""

# by_epd never needs more than a handful of hits, and a very early,
# heavily-transposed position (e.g. after 1.e4) can be the prefix of
# thousands of named lines - cap it so that lookup stays cheap.
_EPD_LOOKUP_LIMIT = 30

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Common function words dropped from BM25 queries - an OR query over every
# token in a natural-language motif description otherwise lets "the" or
# "against" match nearly any chunk in the corpus, drowning out real hits.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "of", "at", "by", "for", "with", "against",
    "to", "from", "in", "on", "into", "onto", "it", "its", "this", "that",
    "as", "not", "no", "so", "can", "will", "would", "has", "have", "had",
}


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        source=row["source"],
        title=row["title"],
        text=row["text"],
        eco=row["eco"],
        epd=row["epd"],
        tags=json.loads(row["tags"]),
        url=row["url"],
        extra_epds=[],  # not reconstructed on read - epd_index isn't part of the Chunk shape
    )


def _fts_query(raw_query: str) -> str:
    """Turn free text into a safe FTS5 MATCH query - quote each alnum token
    and OR them together, so punctuation in motif tags (":", "+", "->")
    can't be misread as FTS5 query syntax.
    """
    tokens = [t for t in _TOKEN_RE.findall(raw_query) if len(t) > 2 and t.lower() not in _STOPWORDS]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


class CorpusStore:
    def __init__(self, db_path: Path | str):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CorpusStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def rebuild(self) -> None:
        self._conn.executescript(
            "DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS chunks_fts; DROP TABLE IF EXISTS epd_index;"
            + _SCHEMA
        )
        self._conn.commit()

    def add_chunk(self, chunk: Chunk) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO chunks (id, source, title, text, eco, epd, tags, url) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (chunk.id, chunk.source, chunk.title, chunk.text, chunk.eco, chunk.epd, json.dumps(chunk.tags), chunk.url),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO chunks_fts (rowid, id, title, text) "
            "SELECT rowid, id, title, text FROM chunks WHERE id = ?",
            (chunk.id,),
        )
        epds = set(chunk.extra_epds)
        if chunk.epd:
            epds.add(chunk.epd)
        self._conn.executemany(
            "INSERT OR REPLACE INTO epd_index (epd, chunk_id) VALUES (?, ?)",
            [(epd, chunk.id) for epd in epds],
        )

    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        n = 0
        for chunk in chunks:
            self.add_chunk(chunk)
            n += 1
        return n

    def commit(self) -> None:
        self._conn.commit()

    def get(self, chunk_id: str) -> Chunk | None:
        row = self._conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        return _row_to_chunk(row) if row else None

    def by_epd(self, epd: str) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT chunks.* FROM epd_index JOIN chunks ON chunks.id = epd_index.chunk_id "
            "WHERE epd_index.epd = ? LIMIT ?",
            (epd, _EPD_LOOKUP_LIMIT),
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def by_eco_prefix(self, eco: str) -> list[Chunk]:
        """ECO family match - 'B12' matches 'B12' exactly, 'B1' matches the
        whole B10-B19 family."""
        rows = self._conn.execute("SELECT * FROM chunks WHERE eco LIKE ?", (eco + "%",)).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def by_tags(self, tags: list[str], limit: int = 20) -> list[Chunk]:
        if not tags:
            return []
        rows = self._conn.execute("SELECT * FROM chunks").fetchall()
        matches = []
        wanted = set(tags)
        for row in rows:
            row_tags = set(json.loads(row["tags"]))
            if row_tags & wanted:
                matches.append((_row_to_chunk(row), len(row_tags & wanted)))
        matches.sort(key=lambda pair: pair[1], reverse=True)
        return [c for c, _ in matches[:limit]]

    def bm25_search(self, query: str, limit: int = 20) -> list[tuple[Chunk, float]]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        rows = self._conn.execute(
            "SELECT chunks.*, bm25(chunks_fts) AS score FROM chunks_fts "
            "JOIN chunks ON chunks.id = chunks_fts.id "
            "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
            (fts_query, limit),
        ).fetchall()
        # sqlite's bm25() returns lower-is-better; flip sign so callers can
        # treat higher as more relevant, matching every other channel here.
        return [(_row_to_chunk(r), -r["score"]) for r in rows]

    def delete_by_source(self, source: str) -> None:
        ids = [r["id"] for r in self._conn.execute("SELECT id FROM chunks WHERE source = ?", (source,)).fetchall()]
        for chunk_id in ids:
            self._conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
            self._conn.execute("DELETE FROM chunks_fts WHERE id = ?", (chunk_id,))
            self._conn.execute("DELETE FROM epd_index WHERE chunk_id = ?", (chunk_id,))
        self._conn.commit()

    def counts(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT source, COUNT(*) AS n FROM chunks GROUP BY source").fetchall()
        return {r["source"]: r["n"] for r in rows}

    def total(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
