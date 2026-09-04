"""Static, non-tactical position features: material, phase, pawn skeleton,
king safety, mobility, and outposts. These don't depend on any candidate
move - they describe the position as it stands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

from chesslens.features.see import PIECE_VALUES

# Standard 24-point phase scale: 4 minors*1 + 4 rooks*2 + 2 queens*4, per side doubled.
_PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
_PHASE_TOTAL = 24


@dataclass
class PawnFlags:
    isolated: list[str] = field(default_factory=list)
    doubled: list[str] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)
    backward: list[str] = field(default_factory=list)


@dataclass
class StaticFeatures:
    fen: str
    phase: str  # "opening" | "middlegame" | "endgame"
    phase_fraction: float  # 0.0 (endgame) .. 1.0 (all material on)
    material: dict[str, int]  # {"white": cp, "black": cp}
    material_diff: int  # white - black, centipawns
    has_bishop_pair: dict[str, bool]
    mobility: dict[str, int]  # legal-move count per side, if it were their turn
    king_safety: dict[str, float]  # heuristic score, higher = safer, per side
    pawns: dict[str, PawnFlags]
    outposts: list[str]  # squares (either color) holding an outpost knight/bishop


def _material(board: chess.Board) -> tuple[dict[str, int], int]:
    material = {"white": 0, "black": 0}
    for piece_type in PIECE_VALUES:
        if piece_type == chess.KING:
            continue
        material["white"] += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
        material["black"] += len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
    return material, material["white"] - material["black"]


def _phase(board: chess.Board) -> tuple[str, float]:
    weight = 0
    for piece_type, w in _PHASE_WEIGHTS.items():
        weight += len(board.pieces(piece_type, chess.WHITE)) * w
        weight += len(board.pieces(piece_type, chess.BLACK)) * w
    fraction = min(1.0, weight / _PHASE_TOTAL)
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    if fraction >= 0.85:
        label = "opening"
    elif fraction <= 0.35 or (queens == 0 and fraction <= 0.55):
        label = "endgame"
    else:
        label = "middlegame"
    return label, fraction


def _mobility(board: chess.Board) -> dict[str, int]:
    result = {}
    for color, key in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        probe = board.copy(stack=False)
        probe.turn = color
        probe.clear_stack()
        result[key] = probe.legal_moves.count()
    return result


def _king_zone(board: chess.Board, color: chess.Color) -> set[chess.Square]:
    king_sq = board.king(color)
    if king_sq is None:
        return set()
    zone = {king_sq}
    zone.update(chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]))
    return zone


def _king_safety(board: chess.Board, color: chess.Color) -> float:
    """Higher is safer. Rewards a pawn shield in front of the king and
    penalizes open files/diagonals into the king zone and enemy attackers
    already covering it. Not a full eval term, just enough to compare
    "this move weakened the king" across a diff.
    """
    king_sq = board.king(color)
    if king_sq is None:
        return 0.0
    zone = _king_zone(board, color)
    forward = 1 if color == chess.WHITE else -1
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)

    shield = 0
    for df in (-1, 0, 1):
        f = king_file + df
        if not 0 <= f <= 7:
            continue
        r = king_rank + forward
        if not 0 <= r <= 7:
            continue
        sq = chess.square(f, r)
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.PAWN and piece.color == color:
            shield += 1

    enemy = not color
    attackers = sum(len(board.attackers(enemy, sq)) for sq in zone)

    open_files_near_king = 0
    for df in (-1, 0, 1):
        f = king_file + df
        if not 0 <= f <= 7:
            continue
        file_has_own_pawn = any(
            board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, color) for r in range(8)
        )
        if not file_has_own_pawn:
            open_files_near_king += 1

    score = 10.0 + shield * 3.0 - attackers * 2.5 - open_files_near_king * 2.0
    return round(score, 2)


def _file_has_pawn(board: chess.Board, file_idx: int, color: chess.Color) -> bool:
    return any(
        board.piece_at(chess.square(file_idx, r)) == chess.Piece(chess.PAWN, color) for r in range(8)
    )


def _pawn_flags(board: chess.Board, color: chess.Color) -> PawnFlags:
    flags = PawnFlags()
    enemy = not color
    forward = 1 if color == chess.WHITE else -1
    pawn_squares = list(board.pieces(chess.PAWN, color))
    by_file: dict[int, list[int]] = {}
    for sq in pawn_squares:
        by_file.setdefault(chess.square_file(sq), []).append(chess.square_rank(sq))

    for sq in pawn_squares:
        name = chess.square_name(sq)
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)

        has_neighbor = _file_has_pawn(board, file_idx - 1, color) or _file_has_pawn(board, file_idx + 1, color)
        if not has_neighbor:
            flags.isolated.append(name)

        if len(by_file[file_idx]) > 1:
            flags.doubled.append(name)

        passed = True
        for df in (-1, 0, 1):
            f = file_idx + df
            if not 0 <= f <= 7:
                continue
            for r in range(8):
                if forward == 1 and r <= rank_idx:
                    continue
                if forward == -1 and r >= rank_idx:
                    continue
                if board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, enemy):
                    passed = False
                    break
            if not passed:
                break
        if passed:
            flags.passed.append(name)

        behind_support = False
        for df in (-1, 1):
            f = file_idx + df
            if not 0 <= f <= 7:
                continue
            for r in range(8):
                if forward == 1 and r >= rank_idx:
                    continue
                if forward == -1 and r <= rank_idx:
                    continue
                if board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, color):
                    behind_support = True
        stop_square = chess.square(file_idx, rank_idx + forward) if 0 <= rank_idx + forward <= 7 else None
        stop_contested = stop_square is not None and bool(board.attackers(enemy, stop_square))
        if not behind_support and stop_contested:
            flags.backward.append(name)

    return flags


def _is_outpost(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    piece = board.piece_at(square)
    if piece is None or piece.color != color or piece.piece_type not in (chess.KNIGHT, chess.BISHOP):
        return False
    enemy = not color
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    forward = 1 if color == chess.WHITE else -1

    supported = False
    for df in (-1, 1):
        f = file_idx + df
        r = rank_idx - forward
        if 0 <= f <= 7 and 0 <= r <= 7:
            if board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, color):
                supported = True
    if not supported:
        return False

    for df in (-1, 1):
        f = file_idx + df
        if not 0 <= f <= 7:
            continue
        for r in range(8):
            if forward == 1 and r <= rank_idx:
                continue
            if forward == -1 and r >= rank_idx:
                continue
            if board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, enemy):
                return False
    return True


def extract_static_features(board: chess.Board) -> StaticFeatures:
    material, diff = _material(board)
    phase, fraction = _phase(board)

    bishop_pair = {
        "white": len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2,
        "black": len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2,
    }

    outposts = [
        chess.square_name(sq)
        for sq in chess.SQUARES
        if board.piece_at(sq) and _is_outpost(board, sq, board.piece_at(sq).color)
    ]

    return StaticFeatures(
        fen=board.fen(),
        phase=phase,
        phase_fraction=round(fraction, 2),
        material=material,
        material_diff=diff,
        has_bishop_pair=bishop_pair,
        mobility=_mobility(board),
        king_safety={
            "white": _king_safety(board, chess.WHITE),
            "black": _king_safety(board, chess.BLACK),
        },
        pawns={
            "white": _pawn_flags(board, chess.WHITE),
            "black": _pawn_flags(board, chess.BLACK),
        },
        outposts=outposts,
    )
