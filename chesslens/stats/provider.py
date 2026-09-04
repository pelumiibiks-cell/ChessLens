"""The empirical grounding channel: what do strong players actually play in
this exact position, and how does it score? An engine alone can never
answer this - "Stockfish rates Nf3 and Bb5 within 0.1 of each other, but
masters play Bb5 in 78% of games and score better with it" is exactly the
kind of framing a coach can offer that a bare evaluation number can't.

Two providers share this interface: LocalPgnStats (offline, built from a
local PGN collection, always available) and, when a Lichess token is
configured, a rating-band explorer provider that can add "at your level,
41% of players go wrong here" - a more useful signal for this audience than
pure grandmaster statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import chess


@dataclass
class MoveStat:
    san: str
    white_wins: int
    draws: int
    black_wins: int

    @property
    def total(self) -> int:
        return self.white_wins + self.draws + self.black_wins

    def score_percent(self, color: chess.Color) -> float:
        """Score for `color` as a percentage (win=1, draw=0.5, loss=0)."""
        if self.total == 0:
            return 0.0
        wins = self.white_wins if color == chess.WHITE else self.black_wins
        return 100.0 * (wins + 0.5 * self.draws) / self.total


@dataclass
class OpeningStats:
    epd: str
    total_games: int
    moves: list[MoveStat]  # sorted by total games descending

    def summary(self, mover: chess.Color, limit: int = 3) -> str:
        """A short prose line for the explanation prompt, e.g.
        'Nf3 (61%, 1204 games), Bb5 (30%, 592 games)'.
        """
        if not self.moves or self.total_games == 0:
            return ""
        parts = []
        for stat in self.moves[:limit]:
            share = 100.0 * stat.total / self.total_games
            parts.append(f"{stat.san} ({share:.0f}% of games, scores {stat.score_percent(mover):.0f}% for the mover)")
        return f"Across {self.total_games} games in this exact position: " + "; ".join(parts) + "."


class OpeningStatsProvider(Protocol):
    def stats_for(self, board: chess.Board) -> OpeningStats | None: ...
