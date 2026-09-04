"""Pull a Chess.com player's own recent games as review targets. No auth
required - the public API exposes per-month game archives.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "chesslens (https://github.com/, contact via project owner)"


def _fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_chesscom_games(username: str, max_games: int = 20) -> str:
    """Returns the user's most recent games (most recent month(s), capped
    to max_games) concatenated as multi-game PGN text.
    """
    encoded = urllib.parse.quote(username.lower(), safe="")
    try:
        archives = _fetch_json(f"https://api.chess.com/pub/player/{encoded}/games/archives")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ValueError(f"No Chess.com user found: '{username}'") from exc
        raise

    months = list(reversed(archives.get("archives", [])))
    pgns: list[str] = []
    for month_url in months:
        if len(pgns) >= max_games:
            break
        month_data = _fetch_json(month_url)
        for game in reversed(month_data.get("games", [])):
            if len(pgns) >= max_games:
                break
            pgn = game.get("pgn")
            if pgn:
                pgns.append(pgn)

    return "\n\n".join(pgns)
