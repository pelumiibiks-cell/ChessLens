"""A held-out set of positions with hand-verified expected motifs and
classifications, run against the actual pipeline. Reports:
  - motif detection recall/precision (does features/motifs.py find what a
    human would call out in these exact positions?)
  - classification accuracy (does the engine-driven classifier agree with
    the expected label?)
  - verifier pass rate on generated explanations (offline by default; pass
    --live to spend real API calls and get a true measure of the model's
    behavior, not just the offline template's)

Most engine-plus-LLM coaching projects have no number to iterate against.
This gives ChessLens one.

Usage: python scripts/eval.py [--live] [--effort medium]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess

from chesslens.config import get_settings
from chesslens.engine.analysis import Engine
from chesslens.engine.classify import Classification, classify_move
from chesslens.explain.client import ExplainClient, effort_for, offline_explanation
from chesslens.explain.prompts import render_position_brief
from chesslens.features.diff import diff_moves
from chesslens.features.motifs import extract_motifs
from chesslens.verify.checks import verify_explanation


def _board_from(pieces: dict[str, str], turn: chess.Color = chess.WHITE) -> chess.Board:
    board = chess.Board.empty()
    for square_name, symbol in pieces.items():
        color = chess.WHITE if symbol.isupper() else chess.BLACK
        piece_type = chess.Piece.from_symbol(symbol).piece_type
        board.set_piece_at(chess.parse_square(square_name), chess.Piece(piece_type, color))
    board.turn = turn
    return board


@dataclass
class EvalCase:
    name: str
    pieces: dict[str, str]
    move_san: str
    expected_motif_kinds: set[str]
    expected_classification: Classification | None = None  # None = don't check


CASES = [
    EvalCase("fork", {"e1": "K", "c3": "N", "e8": "k", "d6": "q", "f6": "r"}, "Ne4", {"fork"}),
    EvalCase("pin", {"e1": "K", "c4": "B", "e8": "k", "d7": "n"}, "Bb5", {"pin"}),
    EvalCase("skewer", {"e1": "K", "d1": "R", "e8": "k", "d7": "q", "d8": "r"}, "Rd5", {"skewer", "hangs"}),
    EvalCase("discovered_check", {"e1": "K", "d1": "R", "d5": "N", "d8": "k"}, "Nc7+", {"discovered_check", "hangs"}),
    # No classification expectation here: with only K+R vs K+B on the board,
    # losing the rook is still a theoretical draw (K+B alone can't mate), so
    # Stockfish correctly rates every move as equal - the point of this case
    # is motif detection, not classification.
    EvalCase("hangs_own_piece", {"e1": "K", "d5": "R", "e8": "k", "c6": "b"}, "Kd2", {"hangs"}),
    EvalCase("back_rank_mate_threat", {"e1": "K", "a1": "R", "g8": "k", "f7": "p", "g7": "p", "h7": "p"}, "Ra8+", {"back_rank_mate_threat"}),
    EvalCase("trapped", {"e1": "K", "a5": "k", "h8": "n", "c4": "B", "b1": "B"}, "Kf1", {"trapped"}),
    # White's rook on b7 is also genuinely hanging to the rook on d7 along
    # the 7th rank here - a real second fact, not a false positive.
    EvalCase("overload", {"e1": "K", "a8": "k", "e5": "n", "d7": "r", "g6": "p", "b7": "R", "c2": "B"}, "Kf1", {"overload", "hangs"}),
    EvalCase("free_queen_capture", {"e1": "K", "d4": "P", "e8": "k", "e5": "q"}, "Kd1", {"hangs"}, Classification.BLUNDER),
]


@dataclass
class CaseResult:
    case: EvalCase
    motif_recall: float
    motif_precision: float
    classification_match: bool | None
    verify_ok: bool | None = None
    verify_errors: list[str] = field(default_factory=list)


def run_eval(live: bool, effort: str | None) -> list[CaseResult]:
    settings = get_settings()
    client = ExplainClient(settings) if live else None
    results = []

    with Engine(settings) as engine:
        for case in CASES:
            board = _board_from(case.pieces)
            move = board.parse_san(case.move_san)

            found_motifs = extract_motifs(board, move)
            found_kinds = {m.kind for m in found_motifs}
            recall = len(found_kinds & case.expected_motif_kinds) / len(case.expected_motif_kinds) if case.expected_motif_kinds else 1.0
            precision = len(found_kinds & case.expected_motif_kinds) / len(found_kinds) if found_kinds else (1.0 if not case.expected_motif_kinds else 0.0)

            analysis = engine.analyse(board, multipv=settings.engine_multipv)
            _cp, _mate, mover_win_after = engine.eval_after(board, move)
            classification = classify_move(
                board_before=board, analysis_before=analysis, move=move,
                mover_win_percent_after=mover_win_after, thresholds=settings.classification_thresholds,
            )
            classification_match = (
                classification.classification == case.expected_classification
                if case.expected_classification is not None else None
            )

            diff = diff_moves(board, analysis.best.move, move)
            if live:
                brief = render_position_brief(
                    fen=board.fen(), mover=diff.mover, classification=classification,
                    analysis=analysis, diff=diff, retrieved=[],
                )
                explanation = client.explain(brief, classification.classification, effort=effort or effort_for(classification.classification))
            else:
                explanation = offline_explanation(classification, diff)
            verification = verify_explanation(explanation, board, diff, [], classification)

            results.append(CaseResult(
                case=case, motif_recall=recall, motif_precision=precision,
                classification_match=classification_match,
                verify_ok=verification.ok, verify_errors=verification.errors,
            ))

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Use the real API instead of offline templates.")
    parser.add_argument("--effort", default=None, help="Override thinking effort for --live runs.")
    args = parser.parse_args()

    results = run_eval(live=args.live, effort=args.effort)

    print(f"{'case':<24} {'motif recall':>12} {'precision':>10} {'class ok':>9} {'verify ok':>10}")
    for r in results:
        class_str = "-" if r.classification_match is None else ("yes" if r.classification_match else "NO")
        verify_str = "-" if r.verify_ok is None else ("yes" if r.verify_ok else "NO")
        print(f"{r.case.name:<24} {r.motif_recall:>11.0%} {r.motif_precision:>9.0%} {class_str:>9} {verify_str:>10}")
        if r.verify_errors:
            for err in r.verify_errors:
                print(f"    ! {err}")

    n = len(results)
    avg_recall = sum(r.motif_recall for r in results) / n
    avg_precision = sum(r.motif_precision for r in results) / n
    class_checked = [r for r in results if r.classification_match is not None]
    class_acc = sum(r.classification_match for r in class_checked) / len(class_checked) if class_checked else float("nan")
    verify_pass_rate = sum(1 for r in results if r.verify_ok) / n

    print()
    print(f"mean motif recall:    {avg_recall:.0%}")
    print(f"mean motif precision: {avg_precision:.0%}")
    print(f"classification accuracy: {class_acc:.0%} ({len(class_checked)} cases checked)")
    print(f"verifier pass rate:    {verify_pass_rate:.0%}")
    print(f"mode: {'live API' if args.live else 'offline template'}")


if __name__ == "__main__":
    main()
