"""Static exchange evaluation.

python-chess has no SEE built in. This is the swap algorithm from the chess
programming wiki, adapted to walk a mutable board copy: instead of tracking
bitboards by hand, each step removes the capturing piece from its origin
square and lets `Board.attackers()` recompute from the mutated occupancy -
that recomputation is what makes x-ray attacks (a rook behind the pawn that
just captured, etc.) fall out for free.
"""

from __future__ import annotations

import chess

PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}


def _least_valuable_attacker(board: chess.Board, attackers: chess.SquareSet) -> chess.Square:
    return min(attackers, key=lambda sq: PIECE_VALUES[board.piece_at(sq).piece_type])


def static_exchange_eval(board: chess.Board, move: chess.Move) -> int:
    """Net material (centipawns) the mover gains from the full capture
    sequence on `move.to_square`, assuming both sides recapture with their
    least valuable piece and play optimally (stop when it's no longer
    favorable). Returns 0 for non-capturing moves.
    """
    if not board.is_capture(move):
        return 0

    board = board.copy(stack=False)
    to_sq = move.to_square
    from_sq = move.from_square
    mover_color = board.turn

    if board.is_en_passant(move):
        captured_value = PIECE_VALUES[chess.PAWN]
        ep_square = to_sq + (-8 if mover_color == chess.WHITE else 8)
    else:
        captured = board.piece_at(to_sq)
        captured_value = PIECE_VALUES[captured.piece_type] if captured else 0
        ep_square = None

    attacker = board.piece_at(from_sq)
    if attacker is None:
        return 0
    attacker_type = attacker.piece_type

    gains = [captured_value]
    board.remove_piece_at(from_sq)
    if ep_square is not None:
        board.remove_piece_at(ep_square)

    side = not mover_color
    value_on_square = PIECE_VALUES[attacker_type]

    while True:
        attackers = board.attackers(side, to_sq)
        if not attackers:
            break
        least_sq = _least_valuable_attacker(board, attackers)
        least_type = board.piece_at(least_sq).piece_type
        gains.append(value_on_square - gains[-1])
        board.remove_piece_at(least_sq)
        value_on_square = PIECE_VALUES[least_type]
        side = not side

    for i in range(len(gains) - 2, -1, -1):
        gains[i] = -max(-gains[i], gains[i + 1])
    return gains[0]


def hanging_value(board: chess.Board, square: chess.Square, attacker_color: chess.Color | None = None) -> int | None:
    """SEE gain for `attacker_color` (default: the side to move) if it
    captures the piece on `square` with its cheapest attacker, or None if
    `square` isn't attacked at all by that color. Takes an explicit color
    rather than always reading board.turn so callers can ask "is this
    hanging to White" on a board where it's Black to move - e.g. checking
    whether the side that just moved left a follow-up capture available.
    """
    attacker_color = board.turn if attacker_color is None else attacker_color
    piece = board.piece_at(square)
    if piece is None or piece.color == attacker_color:
        return None
    attackers = board.attackers(attacker_color, square)
    if not attackers:
        return None
    attacker_sq = _least_valuable_attacker(board, attackers)

    # static_exchange_eval always evaluates from board.turn's perspective, so
    # give it a same-position board where that's actually attacker_color.
    probe = board if board.turn == attacker_color else board.copy(stack=False)
    if probe is not board:
        probe.turn = attacker_color
        probe.clear_stack()

    move = chess.Move(attacker_sq, square)
    if board.piece_at(attacker_sq).piece_type == chess.PAWN and chess.square_rank(square) in (0, 7):
        move = chess.Move(attacker_sq, square, promotion=chess.QUEEN)
    return static_exchange_eval(probe, move)
