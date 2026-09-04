"""Hybrid retrieval: exact EPD lookup, structural (ECO family / motif and
structure tags), and BM25 full text, fused with reciprocal rank fusion.

The query is a position, not a sentence - "what does theory say about this
exact spot on the board" (EPD), "what's written about this kind of
structure" (structural), and "what's written using this vocabulary"
(BM25) are three different, complementary ways of asking that. RRF just
means: a chunk that shows up near the top of more than one channel outranks
a chunk that only shows up in one, without having to hand-tune weights
between fundamentally different score scales (a BM25 score and "is the EPD
identical" aren't comparable numbers on their own).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

from chesslens.features.motifs import Motif
from chesslens.features.structures import PawnStructure
from chesslens.rag.schema import Chunk
from chesslens.rag.store import CorpusStore

RRF_K = 60


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    channels: list[str] = field(default_factory=list)


def _rrf_accumulate(scores: dict[str, float], channel_hits: dict[str, list[str]], channel_name: str, ranked: list[Chunk]) -> None:
    for rank, chunk in enumerate(ranked):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (RRF_K + rank + 1)
        channel_hits.setdefault(chunk.id, []).append(channel_name)


def _dedupe(chunks: list[Chunk]) -> list[Chunk]:
    seen = set()
    out = []
    for c in chunks:
        if c.id in seen:
            continue
        seen.add(c.id)
        out.append(c)
    return out


def retrieve(
    store: CorpusStore,
    board: chess.Board,
    motifs: list[Motif] | None = None,
    structures: list[PawnStructure] | None = None,
    eco: str | None = None,
    free_text: str | None = None,
    k: int = 6,
) -> list[RetrievalResult]:
    motifs = motifs or []
    structures = structures or []

    epd_matches = store.by_epd(board.epd())

    structural_tags = [m.kind for m in motifs] + [s.name for s in structures]
    structural_matches = store.by_tags(structural_tags, limit=20)
    if eco:
        structural_matches = _dedupe(store.by_eco_prefix(eco) + structural_matches)

    query_parts = [eco or ""]
    query_parts += [m.kind for m in motifs] + [m.description for m in motifs]
    query_parts += [s.name for s in structures] + [s.description for s in structures]
    if free_text:
        query_parts.append(free_text)
    query_text = " ".join(p for p in query_parts if p)
    bm25_matches = [c for c, _ in store.bm25_search(query_text, limit=20)] if query_text.strip() else []

    scores: dict[str, float] = {}
    channel_hits: dict[str, list[str]] = {}
    _rrf_accumulate(scores, channel_hits, "epd", epd_matches)
    _rrf_accumulate(scores, channel_hits, "structural", structural_matches)
    _rrf_accumulate(scores, channel_hits, "bm25", bm25_matches)

    all_chunks = {c.id: c for c in epd_matches + structural_matches + bm25_matches}
    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    return [
        RetrievalResult(chunk=all_chunks[cid], score=scores[cid], channels=channel_hits[cid])
        for cid in ranked_ids[:k]
    ]
