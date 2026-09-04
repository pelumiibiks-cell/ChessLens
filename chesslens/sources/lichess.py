"""Pull a Lichess player's own recent games as review targets. No token
required - the games export endpoint is public.
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "chesslens (https://github.com/, contact via project owner)"


def fetch_lichess_games(username: str, max_games: int = 20) -> str:
    """Returns raw multi-game PGN text of the user's most recent games."""
    encoded = urllib.parse.quote(username, safe="")
    url = f"https://lichess.org/api/games/user/{encoded}?max={max_games}&evals=false&clocks=false"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/x-chess-pgn"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ValueError(f"No Lichess user found: '{username}'") from exc
        raise
