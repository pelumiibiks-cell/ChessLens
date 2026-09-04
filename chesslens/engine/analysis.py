"""Stockfish wrapper: multipv analysis and win-percent conversion.

Evals are always reported as centipawns from the perspective of the side to
move in the position being analysed (python-chess's PovScore.relative), and
converted to a win probability with the Lichess sigmoid so that downstream
classification works in "percentage points of win chance", not raw cp - a
drop from +900 to +700 barely changes the outcome, but a drop from +50 to
-150 does, and cp-delta thresholds treat those backwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import chess
import chess.engine

from chesslens.config import Settings, get_settings

MATE_SCORE_CP = 100_000


def cp_to_win_percent(cp: float) -> float:
    """Lichess's cp -> win% formula. 50.0 is an even position."""
    return 100.0 / (1.0 + math.exp(-0.00368208 * cp))


@dataclass
class AnalysisLine:
    move: chess.Move
    san: str
    pv: list[chess.Move]
    cp: int  # from the perspective of the side to move; mate scores saturate this
    mate: int | None  # positive = side to move mates in N, negative = gets mated in N
    win_percent: float


@dataclass
class PositionAnalysis:
    fen: str
    lines: list[AnalysisLine]  # best first

    @property
    def best(self) -> AnalysisLine:
        return self.lines[0]

    def line_for(self, move: chess.Move) -> AnalysisLine | None:
        for line in self.lines:
            if line.move == move:
                return line
        return None

    def pv_san(self, line: AnalysisLine, limit: int = 6) -> list[str]:
        board = chess.Board(self.fen)
        sans = []
        for move in line.pv[:limit]:
            if move not in board.legal_moves:
                break
            sans.append(board.san(move))
            board.push(move)
        return sans


def _score_to_cp_mate(pov_score: chess.engine.PovScore) -> tuple[int, int | None]:
    relative = pov_score.relative
    mate = relative.mate()
    cp = relative.score(mate_score=MATE_SCORE_CP)
    return cp, mate


class Engine:
    """Reusable Stockfish process. Use as a context manager."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if self.settings.stockfish_path is None:
            raise RuntimeError(
                "No Stockfish binary found. Run `python scripts/setup_stockfish.py` "
                "or set STOCKFISH_PATH in .env."
            )
        try:
            # Generous handshake timeout: on Windows, antivirus on-access
            # scanning of a freshly downloaded engine binary can add several
            # seconds to the first few launches, easily exceeding
            # python-chess's 10s default.
            self._engine = chess.engine.SimpleEngine.popen_uci(str(self.settings.stockfish_path), timeout=45.0)
        except OSError as exc:
            # A configured-but-wrong STOCKFISH_PATH fails here with a raw,
            # unhelpful OSError from the subprocess spawn - normalize it to
            # match the "not configured at all" case above.
            raise RuntimeError(
                f"Failed to start Stockfish at '{self.settings.stockfish_path}': {exc}. "
                "Run `python scripts/setup_stockfish.py` or fix STOCKFISH_PATH in .env."
            ) from exc

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        self._engine.quit()

    def analyse(
        self,
        board: chess.Board,
        multipv: int | None = None,
        movetime_ms: int | None = None,
    ) -> PositionAnalysis:
        """Full multipv analysis of `board`, best line first."""
        multipv = multipv or self.settings.engine_multipv
        movetime_ms = movetime_ms or self.settings.engine_depth_ms
        legal_count = board.legal_moves.count()
        multipv = max(1, min(multipv, legal_count)) if legal_count else 1

        infos = self._engine.analyse(
            board, chess.engine.Limit(time=movetime_ms / 1000), multipv=multipv
        )
        if isinstance(infos, dict):
            infos = [infos]

        lines: list[AnalysisLine] = []
        for info in infos:
            pv = info.get("pv") or []
            if not pv:
                continue
            move = pv[0]
            cp, mate = _score_to_cp_mate(info["score"])
            lines.append(
                AnalysisLine(
                    move=move,
                    san=board.san(move),
                    pv=list(pv),
                    cp=cp,
                    mate=mate,
                    win_percent=cp_to_win_percent(cp),
                )
            )
        lines.sort(key=lambda line: line.cp, reverse=True)
        return PositionAnalysis(fen=board.fen(), lines=lines)

    def eval_after(self, board: chess.Board, move: chess.Move, movetime_ms: int | None = None) -> tuple[int, int | None, float]:
        """Eval of the position after `move`, translated back to the mover's
        perspective (negate the resulting side-to-move's score). Returns
        (cp, mate, win_percent) for the player who just moved.
        """
        after = board.copy(stack=False)
        after.push(move)
        if after.is_checkmate():
            return -MATE_SCORE_CP, -1, 100.0  # mover delivered mate
        movetime_ms = movetime_ms or self.settings.engine_depth_ms
        info = self._engine.analyse(after, chess.engine.Limit(time=movetime_ms / 1000))
        cp, mate = _score_to_cp_mate(info["score"])
        mover_cp = -cp
        mover_mate = -mate if mate is not None else None
        return mover_cp, mover_mate, cp_to_win_percent(mover_cp)
