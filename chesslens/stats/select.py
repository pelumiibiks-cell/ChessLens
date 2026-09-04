"""Picks the right empirical-channel provider: LocalPgnStats always works
offline; when a LICHESS_TOKEN is configured, rating-band explorer data is
tried first (more relevant to this audience - "at your level" beats "what
grandmasters do") and falls back to local stats for positions the explorer
doesn't have data for, or on any request failure.
"""

from __future__ import annotations

import chess

from chesslens.config import STATS_DB, Settings
from chesslens.stats.lichess_explorer import LichessExplorerStats
from chesslens.stats.local_pgn import LocalPgnStats
from chesslens.stats.provider import OpeningStats


class CompositeStats:
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    def stats_for(self, board: chess.Board) -> OpeningStats | None:
        if self.primary is not None:
            result = self.primary.stats_for(board)
            if result is not None:
                return result
        return self.fallback.stats_for(board) if self.fallback is not None else None

    def close(self) -> None:
        for provider in (self.primary, self.fallback):
            close = getattr(provider, "close", None)
            if close is not None:
                close()


def get_stats_provider(settings: Settings):
    local = LocalPgnStats(STATS_DB) if STATS_DB.is_file() else None
    if settings.lichess_token:
        return CompositeStats(LichessExplorerStats(settings.lichess_token), local)
    return local
