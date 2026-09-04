"""Named pawn-structure archetypes: IQP, Carlsbad, Maroczy Bind, hanging
pawns, Stonewall, King's Indian chain. These are the tags the structural
retrieval channel searches on - "this position is an IQP position" pulls in
different theory than "this is Carlsbad", even at a similar material count.

Each detector is a pattern match on the pawn skeleton only. They're heuristic
by nature (real games are messier than the textbook diagram), so more than
one can fire, and none is guaranteed to fire even when a title player would
recognize the structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

FILES = "abcdefgh"


@dataclass
class PawnStructure:
    name: str
    color: str  # "white" | "black" | "shared"
    description: str


def _pawn_files(board: chess.Board, color: chess.Color) -> dict[str, list[int]]:
    by_file: dict[str, list[int]] = {f: [] for f in FILES}
    for sq in board.pieces(chess.PAWN, color):
        by_file[FILES[chess.square_file(sq)]].append(chess.square_rank(sq))
    return by_file


def _has(by_file: dict[str, list[int]], file_letter: str) -> bool:
    return bool(by_file.get(file_letter))


def _on(by_file: dict[str, list[int]], file_letter: str, rank_idx: int) -> bool:
    return rank_idx in by_file.get(file_letter, [])


def _detect_iqp(white: dict, black: dict) -> list[PawnStructure]:
    found = []
    if _on(white, "d", 3) and not _has(white, "c") and not _has(white, "e"):
        found.append(PawnStructure(
            "isolated_queen_pawn", "white",
            "White has an isolated pawn on d4 - open c- and e-files, active piece play "
            "against a target that becomes a long-term endgame weakness.",
        ))
    if _on(black, "d", 4) and not _has(black, "c") and not _has(black, "e"):
        found.append(PawnStructure(
            "isolated_queen_pawn", "black",
            "Black has an isolated pawn on d5 - open c- and e-files, active piece play "
            "against a target that becomes a long-term endgame weakness.",
        ))
    return found


def _detect_carlsbad(white: dict, black: dict) -> list[PawnStructure]:
    if (
        _has(white, "d") and _has(black, "d")
        and not _has(white, "c") and not _has(white, "e")
        and not _has(black, "c") and not _has(black, "e")
    ):
        return [PawnStructure(
            "carlsbad", "shared",
            "Carlsbad structure (symmetric d-pawns, open c- and e-files for both sides): "
            "White's plan is a minority attack with b4-b5; Black looks for central play or "
            "kingside expansion.",
        )]
    return []


def _detect_maroczy(white: dict, black: dict) -> list[PawnStructure]:
    if _on(white, "c", 3) and _on(white, "e", 3) and not _has(white, "d"):
        return [PawnStructure(
            "maroczy_bind", "white",
            "Maroczy Bind (White pawns on c4 and e4, no d-pawn): White restrains Black's "
            "...d5 and ...b5 breaks and plays for slow central control.",
        )]
    if _on(black, "c", 4) and _on(black, "e", 4) and not _has(black, "d"):
        return [PawnStructure(
            "maroczy_bind", "black",
            "Reversed Maroczy Bind (Black pawns on c5 and e5, no d-pawn): Black restrains "
            "White's central breaks.",
        )]
    return []


def _detect_hanging_pawns(white: dict, black: dict) -> list[PawnStructure]:
    found = []
    for a, b in zip(FILES, FILES[1:]):
        left_flank = FILES[FILES.index(a) - 1] if FILES.index(a) > 0 else None
        right_flank = FILES[FILES.index(b) + 1] if FILES.index(b) < 7 else None
        for color_name, by_file, advanced_rank in (("white", white, 3), ("black", black, 4)):
            ranks_a, ranks_b = by_file.get(a, []), by_file.get(b, [])
            if len(ranks_a) != 1 or len(ranks_b) != 1 or ranks_a[0] != ranks_b[0]:
                continue
            if ranks_a[0] != advanced_rank:
                continue  # home-rank pawns with empty neighboring files aren't "hanging"
            if left_flank and _has(by_file, left_flank):
                continue
            if right_flank and _has(by_file, right_flank):
                continue
            found.append(PawnStructure(
                "hanging_pawns", color_name,
                f"Hanging pawns on {a}{ranks_a[0] + 1} and {b}{ranks_b[0] + 1}: mobile and "
                "space-gaining, but fixed targets once blockaded or attacked from the side.",
            ))
    return found


def _detect_stonewall(white: dict, black: dict) -> list[PawnStructure]:
    found = []
    if _on(white, "d", 3) and _on(white, "e", 2) and _on(white, "f", 3):
        found.append(PawnStructure(
            "stonewall", "white",
            "Stonewall skeleton (pawns on d4/e3/f4): locks the center, aims a piece attack "
            "at the kingside dark squares, but weakens e4 and the a1-h8 diagonal.",
        ))
    if _on(black, "d", 4) and _on(black, "e", 5) and _on(black, "f", 4):
        found.append(PawnStructure(
            "stonewall", "black",
            "Stonewall skeleton (pawns on d5/e6/f5): locks the center, aims a piece attack "
            "at the kingside dark squares, but weakens e5 and the a8-h1 diagonal.",
        ))
    return found


def _detect_kid_chain(white: dict, black: dict) -> list[PawnStructure]:
    if _on(white, "d", 3) and _on(white, "e", 3) and _on(black, "d", 5) and _on(black, "e", 4):
        return [PawnStructure(
            "kings_indian_chain", "shared",
            "Locked King's Indian-style center (White d4/e4 vs Black d6/e5): play shifts to "
            "the wings - White for c5 or f5, Black for ...f5 kingside expansion.",
        )]
    if _on(black, "d", 4) and _on(black, "e", 4) and _on(white, "d", 2) and _on(white, "e", 3):
        return [PawnStructure(
            "kings_indian_chain", "shared",
            "Locked King's Indian-style center, colors reversed (Black d5/e5 vs White d3/e4): "
            "play shifts to the wings.",
        )]
    return []


def classify_pawn_structure(board: chess.Board) -> list[PawnStructure]:
    white = _pawn_files(board, chess.WHITE)
    black = _pawn_files(board, chess.BLACK)
    found: list[PawnStructure] = []
    found += _detect_iqp(white, black)
    found += _detect_carlsbad(white, black)
    found += _detect_maroczy(white, black)
    found += _detect_hanging_pawns(white, black)
    found += _detect_stonewall(white, black)
    found += _detect_kid_chain(white, black)
    return found
