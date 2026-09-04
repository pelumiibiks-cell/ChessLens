"""Move classification from win-percent loss, not raw centipawns.

A drop from +900 to +700 cp is still a completely winning position; a drop
from +50 to -150 cp flips who's winning. Thresholds below are in win%
points, which treats those cases correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import chess

from chesslens.engine.analysis import PositionAnalysis

# Below this win%-loss, a move is indistinguishable from the engine's best.
BEST_TOLERANCE = 0.5


class Classification(str, Enum):
    # classify_move() below only ever returns the values from ONLY_MOVE
    # onward - BOOK is assigned separately by review.py after an opening
    # corpus lookup, which the engine has no way to know about. It lives
    # here anyway so there's exactly one Classification type used
    # everywhere (engine, explain schema, review), not two enums that
    # happen to share string values.
    BOOK = "book"
    ONLY_MOVE = "only_move"
    BEST = "best"
    GOOD = "good"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    BLUNDER = "blunder"


@dataclass
class ClassificationResult:
    classification: Classification
    win_percent_loss: float  # always >= 0


def classify_move(
    *,
    board_before: chess.Board,
    analysis_before: PositionAnalysis,
    move: chess.Move,
    mover_win_percent_after: float,
    thresholds: dict[str, float],
) -> ClassificationResult:
    if board_before.legal_moves.count() == 1:
        return ClassificationResult(Classification.ONLY_MOVE, 0.0)

    best_win = analysis_before.best.win_percent
    win_loss = max(0.0, best_win - mover_win_percent_after)

    if move == analysis_before.best.move or win_loss < BEST_TOLERANCE:
        return ClassificationResult(Classification.BEST, win_loss)
    if win_loss >= thresholds["blunder"]:
        return ClassificationResult(Classification.BLUNDER, win_loss)
    if win_loss >= thresholds["mistake"]:
        return ClassificationResult(Classification.MISTAKE, win_loss)
    if win_loss >= thresholds["inaccuracy"]:
        return ClassificationResult(Classification.INACCURACY, win_loss)
    return ClassificationResult(Classification.GOOD, win_loss)
