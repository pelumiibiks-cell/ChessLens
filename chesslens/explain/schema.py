"""The model's output contract. Every field beyond the prose ones exists so
verify/checks.py has something concrete to check the explanation against -
that's what keeps this "explainable" instead of merely fluent. motifs_cited
must be a subset of what features/motifs.py actually extracted for this
position; moves_cited must all be legal; sources must be real corpus
citation ids.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from chesslens.engine.classify import Classification

__all__ = ["Classification", "Explanation"]


class Explanation(BaseModel):
    headline: str = Field(description="One line: what actually happened with this move.")
    classification: Classification
    why_engine_prefers: str = Field(description="The objective case for the engine's best move.")
    what_your_move_misses: str | None = Field(
        default=None, description="What the played move gives up, compared to best. Null if the played move was best."
    )
    what_masters_do: str | None = Field(
        default=None, description="What the empirical/master-game data shows for this position. Null if off-book or no data."
    )
    principle: str = Field(description="The transferable idea behind this position, in plain language.")
    next_time: str = Field(description="A concrete pattern to look for in a future game.")
    motifs_cited: list[str] = Field(
        default_factory=list, description="Motif tags referenced in the explanation - must be a subset of the extracted tags provided."
    )
    moves_cited: list[str] = Field(
        default_factory=list, description="Every move (in SAN) mentioned in the explanation, for legality verification."
    )
    sources: list[str] = Field(
        default_factory=list, description="Citation ids (from the provided retrieved sources) actually drawn on."
    )
