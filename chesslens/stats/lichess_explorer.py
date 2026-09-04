"""Optional rating-band statistics via the Lichess opening explorer -
"at 1600, 41% of players go wrong here" is a better coaching signal for
this audience than pure grandmaster statistics, but the explorer API
requires a token (a 401 with an empty body otherwise), so this only
activates when LICHESS_TOKEN is set in .env. Falls back to LocalPgnStats
whenever it isn't configured, or on any request failure - a broken network
call here should never take down an explanation.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

import chess

from chesslens.stats.provider import MoveStat, OpeningStats

API_URL = "https://explorer.lichess.ovh/lichess"
USER_AGENT = "chesslens (https://github.com/, contact via project owner)"


class LichessExplorerStats:
    def __init__(self, token: str, ratings: str = "1600,1800,2000", speeds: str = "blitz,rapid,classical", timeout: int = 10):
        self.token = token
        self.ratings = ratings
        self.speeds = speeds
        self.timeout = timeout

    def stats_for(self, board: chess.Board) -> OpeningStats | None:
        fen = board.fen()
        url = f"{API_URL}?fen={urllib.parse.quote(fen)}&ratings={self.ratings}&speeds={self.speeds}"
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Authorization": f"Bearer {self.token}"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None

        raw_moves = data.get("moves", [])
        if not raw_moves:
            return None

        moves = []
        for m in raw_moves:
            san = m.get("san")
            if not san:
                continue
            moves.append(MoveStat(
                san=san,
                white_wins=int(m.get("white", 0)),
                draws=int(m.get("draws", 0)),
                black_wins=int(m.get("black", 0)),
            ))
        if not moves:
            return None
        moves.sort(key=lambda mv: mv.total, reverse=True)
        return OpeningStats(epd=board.epd(), total_games=sum(mv.total for mv in moves), moves=moves)
