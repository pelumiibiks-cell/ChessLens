"""Deterministic verification of a model explanation against the facts it
was given. This is what makes the pipeline "explainable" rather than merely
fluent: every claim the verifier can check mechanically, it does - no
second model call, no judgment call, just set membership and legality.

A failed check doesn't mean the explanation is unusable, only that it made
a claim the pipeline can't stand behind. Callers decide what to do with
that (retry once, strip the offending field, surface a warning).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

from chesslens.engine.classify import Classification, ClassificationResult
from chesslens.explain.schema import Explanation
from chesslens.features.diff import MoveDiff
from chesslens.rag.retrieve import RetrievalResult

ALLOWED_MOTIF_KINDS = {
    "fork", "pin", "skewer", "discovered_attack", "discovered_check", "hangs",
    "back_rank_weakness", "back_rank_mate_threat", "trapped", "overload",
}


@dataclass
class VerificationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def _is_legal_san(board: chess.Board, san: str) -> bool:
    try:
        board.parse_san(san)
        return True
    except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
        return False


def verify_explanation(
    explanation: Explanation,
    board_before: chess.Board,
    diff: MoveDiff,
    retrieved: list[RetrievalResult],
    classification: ClassificationResult,
) -> VerificationResult:
    errors: list[str] = []

    if explanation.classification.value != classification.classification.value:
        errors.append(
            f"classification is '{explanation.classification.value}', but the computed "
            f"classification for this move is '{classification.classification.value}'."
        )

    grounded_kinds = {m.kind for m in diff.best_motifs} | {m.kind for m in diff.played_motifs}
    for tag in explanation.motifs_cited:
        if tag not in ALLOWED_MOTIF_KINDS:
            errors.append(f"motifs_cited references '{tag}', which isn't in the motif vocabulary at all.")
        elif tag not in grounded_kinds:
            errors.append(f"motifs_cited references '{tag}', which wasn't found in this position (best or played line).")

    # A cited move is legitimate if it's legal either in the position being
    # analysed, or one ply later, after the move actually played - coaching
    # language routinely describes a follow-up threat ("...and now Nxe5
    # wins the pawn"), which is only legal in the position the opponent
    # would be moving in next, not the one currently on the board.
    board_after_played = board_before.copy(stack=False)
    try:
        board_after_played.push(board_before.parse_san(diff.played_san))
    except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
        board_after_played = None

    for san in explanation.moves_cited:
        legal_now = _is_legal_san(board_before, san)
        legal_next_ply = board_after_played is not None and _is_legal_san(board_after_played, san)
        if not legal_now and not legal_next_ply:
            errors.append(f"moves_cited includes '{san}', which is not a legal move in this position or the position after {diff.played_san}.")

    retrieved_ids = {r.chunk.id for r in retrieved}
    for source_id in explanation.sources:
        if source_id not in retrieved_ids:
            errors.append(f"sources cites '{source_id}', which wasn't in the retrieved set for this position.")

    # Note: this has to key off the classification, not diff.same_move -
    # classify_move labels a move BEST whenever it's within tolerance of the
    # top engine choice, not only when it's literally the same move object,
    # so a played move can be "as good as best" without diff.same_move
    # being True.
    nothing_missed = classification.classification in (Classification.BEST, Classification.BOOK, Classification.ONLY_MOVE)
    if nothing_missed and explanation.what_your_move_misses is not None:
        errors.append("what_your_move_misses should be null - the played move was as good as the engine's best.")
    if not nothing_missed and explanation.what_your_move_misses is None:
        errors.append("what_your_move_misses is null, but the played move was not the engine's best move.")

    return VerificationResult(ok=not errors, errors=errors)
