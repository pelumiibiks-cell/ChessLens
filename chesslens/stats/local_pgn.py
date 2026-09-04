"""Offline opening statistics built from a local PGN collection. Default
provider - no token, no rate limit, works forever. See scripts/build_stats.py
for how data/stats.db gets populated.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import chess
import chess.pgn

from chesslens.stats.provider import MoveStat, OpeningStats

# Games are only informative for opening/early-middlegame statistics - by
# move 20 nearly every position is unique to that one game, so recording
# deeper plies would just bloat the table with singleton rows that never
# match another position.
DEFAULT_PLY_CAP = 40

_SCHEMA = """
CREATE TABLE IF NOT EXISTS position_stats (
    epd TEXT NOT NULL,
    move_uci TEXT NOT NULL,
    white_wins INTEGER NOT NULL DEFAULT 0,
    draws INTEGER NOT NULL DEFAULT 0,
    black_wins INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (epd, move_uci)
);
CREATE INDEX IF NOT EXISTS idx_position_stats_epd ON position_stats(epd);
"""

_RESULT_COLUMN = {"1-0": "white_wins", "0-1": "black_wins", "1/2-1/2": "draws"}


def build_stats_db(pgn_paths: Iterable[Path], db_path: Path, ply_cap: int = DEFAULT_PLY_CAP) -> int:
    """Replays every game in every PGN file and accumulates per-position
    move counts. Returns the number of games ingested.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript("DROP TABLE IF EXISTS position_stats;" + _SCHEMA)

    games_ingested = 0
    counts: dict[tuple[str, str], list[int]] = {}  # (epd, move_uci) -> [white,draw,black]

    for path in pgn_paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                result_col = _RESULT_COLUMN.get(game.headers.get("Result", "*"))
                if result_col is None:
                    continue  # unclear/ongoing result carries no signal

                board = game.board()
                for i, move in enumerate(game.mainline_moves()):
                    if i >= ply_cap:
                        break
                    epd = board.epd()
                    key = (epd, move.uci())
                    row = counts.setdefault(key, [0, 0, 0])
                    row[["white_wins", "draws", "black_wins"].index(result_col)] += 1
                    board.push(move)
                games_ingested += 1

    conn.executemany(
        "INSERT INTO position_stats (epd, move_uci, white_wins, draws, black_wins) VALUES (?, ?, ?, ?, ?)",
        [(epd, move_uci, w, d, b) for (epd, move_uci), (w, d, b) in counts.items()],
    )
    conn.commit()
    conn.close()
    return games_ingested


class LocalPgnStats:
    def __init__(self, db_path: Path | str):
        self.path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        if self.path.is_file():
            self._conn = sqlite3.connect(str(self.path))
            self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()

    def stats_for(self, board: chess.Board) -> OpeningStats | None:
        if self._conn is None:
            return None
        rows = self._conn.execute(
            "SELECT move_uci, white_wins, draws, black_wins FROM position_stats WHERE epd = ?",
            (board.epd(),),
        ).fetchall()
        if not rows:
            return None

        moves = []
        for row in rows:
            try:
                move = chess.Move.from_uci(row["move_uci"])
                san = board.san(move)
            except (ValueError, chess.IllegalMoveError):
                continue  # stale entry from a different position that happened to share this EPD key
            moves.append(MoveStat(san=san, white_wins=row["white_wins"], draws=row["draws"], black_wins=row["black_wins"]))

        if not moves:
            return None
        moves.sort(key=lambda m: m.total, reverse=True)
        return OpeningStats(epd=board.epd(), total_games=sum(m.total for m in moves), moves=moves)
