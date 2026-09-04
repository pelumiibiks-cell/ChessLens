"""Tactical motif extraction: fork, pin, skewer, discovered attack/check,
hanging pieces, back-rank weakness, trapped pieces, overload.

Every detector here answers a yes/no, verifiable question about the board -
"does the piece on d5 attack two undefended pieces", "is this piece pinned
against its king" - using only python-chess primitives plus the static
exchange evaluator. Nothing here is fuzzy or learned, which is the point:
the tags this module emits are later used as (1) the retrieval query, (2)
the grounding facts handed to the model, and (3) the whitelist a verifier
checks the model's claims against. A tag the model uses that isn't in this
list gets rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

from chesslens.features.see import PIECE_VALUES, hanging_value, static_exchange_eval

MIN_FORK_TARGET_VALUE = PIECE_VALUES[chess.KNIGHT]

_ROOK_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_BISHOP_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


@dataclass
class Motif:
    kind: str  # "fork" | "pin" | "skewer" | "discovered_attack" | "discovered_check"
    #           | "hangs" | "back_rank_weakness" | "back_rank_mate_threat"
    #           | "trapped" | "overload"
    tag: str  # canonical short form, e.g. "fork:Nd5->Qd8,Rf8"
    squares: list[str] = field(default_factory=list)  # algebraic squares involved
    description: str = ""


def _slider_directions(piece_type: chess.PieceType) -> list[tuple[int, int]]:
    if piece_type == chess.BISHOP:
        return _BISHOP_DIRS
    if piece_type == chess.ROOK:
        return _ROOK_DIRS
    if piece_type == chess.QUEEN:
        return _ROOK_DIRS + _BISHOP_DIRS
    return []


def _walk_ray(board: chess.Board, origin: chess.Square, direction: tuple[int, int]) -> list[tuple[chess.Square, chess.Piece]]:
    """Every piece encountered walking from `origin` in `direction` to the
    board edge, in order - unlike attack generation, this does NOT stop
    after the first piece, so it can see what's behind it (needed for
    skewers, pins, and x-ray-style discoveries).
    """
    hits = []
    df, dr = direction
    f, r = chess.square_file(origin) + df, chess.square_rank(origin) + dr
    while 0 <= f <= 7 and 0 <= r <= 7:
        sq = chess.square(f, r)
        piece = board.piece_at(sq)
        if piece is not None:
            hits.append((sq, piece))
        f += df
        r += dr
    return hits


def _direction_between(a: chess.Square, b: chess.Square) -> tuple[int, int] | None:
    fa, ra = chess.square_file(a), chess.square_rank(a)
    fb, rb = chess.square_file(b), chess.square_rank(b)
    df, dr = fb - fa, rb - ra
    if df == 0 and dr == 0:
        return None
    if df != 0 and dr != 0 and abs(df) != abs(dr):
        return None  # not on a straight line
    return ((df > 0) - (df < 0), (dr > 0) - (dr < 0))


def _find_forks(board_after: chess.Board, mover_color: chess.Color, to_sq: chess.Square) -> list[Motif]:
    piece = board_after.piece_at(to_sq)
    if piece is None:
        return []
    targets = []
    for sq in board_after.attacks(to_sq):
        target = board_after.piece_at(sq)
        if target is None or target.color == mover_color:
            continue
        is_king = target.piece_type == chess.KING
        if not is_king and PIECE_VALUES[target.piece_type] < MIN_FORK_TARGET_VALUE:
            continue
        gain = hanging_value(board_after, sq, attacker_color=mover_color)
        threatened = is_king or (gain is not None and gain >= 0)
        if threatened:
            targets.append(sq)
    if len(targets) < 2:
        return []
    target_names = [chess.square_name(s) for s in targets]
    piece_letter = chess.piece_symbol(piece.piece_type).upper()
    origin_name = chess.square_name(to_sq)
    tag = f"fork:{piece_letter}{origin_name}->{','.join(target_names)}"
    return [
        Motif(
            kind="fork",
            tag=tag,
            squares=[origin_name] + target_names,
            description=f"{piece_letter} on {origin_name} forks {', '.join(target_names)}.",
        )
    ]


def _find_pins(board_before: chess.Board, board_after: chess.Board, mover_color: chess.Color) -> list[Motif]:
    enemy = not mover_color
    king_sq = board_after.king(enemy)
    if king_sq is None:
        return []
    motifs = []
    for sq in board_after.pieces(chess.PAWN, enemy) | board_after.pieces(chess.KNIGHT, enemy) | \
              board_after.pieces(chess.BISHOP, enemy) | board_after.pieces(chess.ROOK, enemy) | \
              board_after.pieces(chess.QUEEN, enemy):
        if not board_after.is_pinned(enemy, sq):
            continue
        was_pinned_before = board_before.piece_at(sq) is not None and board_before.is_pinned(enemy, sq)
        if was_pinned_before:
            continue  # not caused by this move

        direction = _direction_between(king_sq, sq)
        if direction is None:
            continue
        hits = _walk_ray(board_after, king_sq, direction)
        pinner_sq = None
        for hit_sq, hit_piece in hits:
            if hit_sq == sq:
                continue
            if hit_piece.color == mover_color and hit_piece.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
                pinner_sq = hit_sq
            break  # only the first non-pinned-piece hit matters, pinned or not

        pinned_piece = board_after.piece_at(sq)
        pinned_letter = chess.piece_symbol(pinned_piece.piece_type).upper()
        sq_name, king_name = chess.square_name(sq), chess.square_name(king_sq)
        if pinner_sq is not None:
            pinner_piece = board_after.piece_at(pinner_sq)
            pinner_letter = chess.piece_symbol(pinner_piece.piece_type).upper()
            pinner_name = chess.square_name(pinner_sq)
            tag = f"pin:{pinner_letter}{pinner_name}->{pinned_letter}{sq_name},K{king_name}"
            desc = f"{pinner_letter} on {pinner_name} pins {pinned_letter} on {sq_name} against the king on {king_name}."
            squares = [pinner_name, sq_name, king_name]
        else:
            tag = f"pin:{pinned_letter}{sq_name},K{king_name}"
            desc = f"{pinned_letter} on {sq_name} is pinned against the king on {king_name}."
            squares = [sq_name, king_name]
        motifs.append(Motif(kind="pin", tag=tag, squares=squares, description=desc))
    return motifs


def _find_skewers(board_after: chess.Board, mover_color: chess.Color, to_sq: chess.Square) -> list[Motif]:
    piece = board_after.piece_at(to_sq)
    if piece is None or piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        return []
    enemy = not mover_color
    motifs = []
    for direction in _slider_directions(piece.piece_type):
        hits = _walk_ray(board_after, to_sq, direction)
        if len(hits) < 2:
            continue
        (front_sq, front_piece), (back_sq, back_piece) = hits[0], hits[1]
        if front_piece.color != enemy or back_piece.color != enemy:
            continue
        if front_piece.piece_type == chess.KING:
            continue  # that's a discovered/direct check line, not a skewer
        if PIECE_VALUES[front_piece.piece_type] < PIECE_VALUES[back_piece.piece_type]:
            continue  # front piece isn't forced to move
        front_name, back_name, origin_name = chess.square_name(front_sq), chess.square_name(back_sq), chess.square_name(to_sq)
        piece_letter = chess.piece_symbol(piece.piece_type).upper()
        front_letter = chess.piece_symbol(front_piece.piece_type).upper()
        back_letter = chess.piece_symbol(back_piece.piece_type).upper()
        tag = f"skewer:{piece_letter}{origin_name}->{front_letter}{front_name},{back_letter}{back_name}"
        motifs.append(
            Motif(
                kind="skewer",
                tag=tag,
                squares=[origin_name, front_name, back_name],
                description=(
                    f"{piece_letter} on {origin_name} skewers {front_letter} on {front_name}, "
                    f"with {back_letter} on {back_name} behind it on the same line."
                ),
            )
        )
    return motifs


def _find_discoveries(board_before: chess.Board, board_after: chess.Board, mover_color: chess.Color, from_sq: chess.Square, to_sq: chess.Square) -> list[Motif]:
    motifs = []
    enemy = not mover_color
    for piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        for source_sq in board_after.pieces(piece_type, mover_color):
            if source_sq == to_sq:
                continue  # that's the moved piece, handled by fork/direct-attack logic
            direction = _direction_between(source_sq, from_sq)
            if direction is None:
                continue
            hits = _walk_ray(board_after, source_sq, direction)
            if not hits:
                continue
            target_sq, target_piece = hits[0]
            if target_piece.color != enemy:
                continue
            if not _on_ray(source_sq, from_sq, target_sq):
                continue
            # Must have actually been blocked at from_sq before the move.
            before_hits = _walk_ray(board_before, source_sq, direction)
            was_blocked_here = bool(before_hits) and before_hits[0][0] == from_sq
            if not was_blocked_here:
                continue

            source_name, target_name = chess.square_name(source_sq), chess.square_name(target_sq)
            source_letter = chess.piece_symbol(piece_type).upper()
            is_check = target_piece.piece_type == chess.KING
            kind = "discovered_check" if is_check else "discovered_attack"
            target_letter = "K" if is_check else chess.piece_symbol(target_piece.piece_type).upper()
            target_word = "king" if is_check else target_letter
            tag = f"{kind}:{source_letter}{source_name}->{target_letter}{target_name}"
            motifs.append(
                Motif(
                    kind=kind,
                    tag=tag,
                    squares=[source_name, target_name],
                    description=f"Moving away uncovers {source_letter} on {source_name}, attacking the {target_word} on {target_name}.",
                )
            )
    return motifs


def _on_ray(source: chess.Square, mid: chess.Square, far: chess.Square) -> bool:
    d1 = _direction_between(source, mid)
    d2 = _direction_between(source, far)
    if d1 is None or d2 is None or d1 != d2:
        return False
    return True


def _find_hanging(board_after: chess.Board, mover_color: chess.Color) -> list[Motif]:
    motifs = []
    for piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
        for sq in board_after.pieces(piece_type, mover_color):
            gain = hanging_value(board_after, sq, attacker_color=not mover_color)
            if gain is not None and gain > 0:
                letter = chess.piece_symbol(piece_type).upper()
                name = chess.square_name(sq)
                motifs.append(
                    Motif(
                        kind="hangs",
                        tag=f"hangs:{letter}{name}",
                        squares=[name],
                        description=f"{letter} on {name} can be won with a favorable capture (net +{gain} cp).",
                    )
                )
    return motifs


def _find_back_rank(board_after: chess.Board, mover_color: chess.Color) -> list[Motif]:
    motifs = []
    for color in (chess.WHITE, chess.BLACK):
        king_sq = board_after.king(color)
        if king_sq is None:
            continue
        back_rank = 0 if color == chess.WHITE else 7
        if chess.square_rank(king_sq) != back_rank:
            continue
        # Gate on actual reachability, not just "no flight square" - every
        # un-castled king at the start of the game has "no escape" on its
        # back rank because its own unmoved pawns block it, but that's not
        # a weakness when no enemy piece can get anywhere near it.
        enemy_reaches_back_rank = any(
            chess.square_rank(sq) == back_rank
            for piece_type in (chess.ROOK, chess.QUEEN)
            for source in board_after.pieces(piece_type, not color)
            for sq in board_after.attacks(source)
        )
        if not enemy_reaches_back_rank:
            continue
        forward = 1 if color == chess.WHITE else -1
        king_file = chess.square_file(king_sq)
        escape_exists = False
        for df in (-1, 0, 1):
            f = king_file + df
            r = back_rank + forward
            if not (0 <= f <= 7 and 0 <= r <= 7):
                continue
            sq = chess.square(f, r)
            occupant = board_after.piece_at(sq)
            if occupant is not None and occupant.color == color:
                continue  # blocked by own piece
            if board_after.attackers(not color, sq):
                continue  # covered by the enemy
            escape_exists = True
        if escape_exists:
            continue
        king_name = chess.square_name(king_sq)

        # Is there an immediate rook/queen move landing on the back rank
        # that gives check? Let python-chess answer that directly instead
        # of hand-rolling line-of-sight logic.
        probe = board_after.copy(stack=False)
        probe.turn = not color
        probe.clear_stack()
        threat = any(
            chess.square_rank(m.to_square) == back_rank
            and probe.piece_at(m.from_square).piece_type in (chess.ROOK, chess.QUEEN)
            and probe.gives_check(m)
            for m in probe.legal_moves
        )
        kind = "back_rank_mate_threat" if threat else "back_rank_weakness"
        motifs.append(
            Motif(
                kind=kind,
                tag=f"{kind}:K{king_name}",
                squares=[king_name],
                description=f"King on {king_name} has no escape square on the back rank.",
            )
        )
    return motifs


def _escape_move_is_unsafe(board_after: chess.Board, move: chess.Move, mover_color: chess.Color) -> bool:
    """Would this destination lose material for the piece's own side, net of
    whatever it captures getting there? A capturing escape (e.g. taking a
    defended piece) can be worth it even if the piece is then recaptured -
    that's exactly what SEE on the move itself already accounts for, so
    captures and quiet moves need different checks.
    """
    if board_after.is_capture(move):
        return static_exchange_eval(board_after, move) < 0
    probe = board_after.copy(stack=False)
    probe.push(move)
    gain = hanging_value(probe, move.to_square, attacker_color=mover_color)
    return gain is not None and gain > 0


def _find_trapped(board_after: chess.Board, mover_color: chess.Color) -> list[Motif]:
    """Enemy pieces (to-move side in board_after) with zero safe destination
    squares - every legal move for that piece lands somewhere the mover can
    win it back.
    """
    enemy = board_after.turn  # side to move in board_after is the mover's opponent
    motifs = []
    for piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        for sq in list(board_after.pieces(piece_type, enemy)):
            destinations = [m for m in board_after.legal_moves if m.from_square == sq]
            if not destinations:
                continue  # can't move at all isn't "trapped" in the tactical sense used here
            all_unsafe = all(_escape_move_is_unsafe(board_after, m, mover_color) for m in destinations)
            if all_unsafe:
                letter = chess.piece_symbol(piece_type).upper()
                name = chess.square_name(sq)
                motifs.append(
                    Motif(
                        kind="trapped",
                        tag=f"trapped:{letter}{name}",
                        squares=[name],
                        description=f"{letter} on {name} has no square that isn't lost to a favorable capture.",
                    )
                )
    return motifs


def _find_overloads(board_after: chess.Board, mover_color: chess.Color) -> list[Motif]:
    enemy = not mover_color
    defended_by: dict[chess.Square, list[chess.Square]] = {}
    for sq in chess.SQUARES:
        piece = board_after.piece_at(sq)
        if piece is None or piece.color != enemy:
            continue
        if not board_after.attackers(mover_color, sq):
            continue  # only assets actually under attack matter
        for defender_sq in board_after.attackers(enemy, sq):
            defended_by.setdefault(defender_sq, []).append(sq)

    motifs = []
    for defender_sq, assets in defended_by.items():
        sole_defense = [
            a for a in assets
            if len(board_after.attackers(enemy, a)) == 1 and defender_sq in board_after.attackers(enemy, a)
        ]
        if len(sole_defense) < 2:
            continue
        defender_piece = board_after.piece_at(defender_sq)
        defender_letter = chess.piece_symbol(defender_piece.piece_type).upper()
        defender_name = chess.square_name(defender_sq)
        asset_names = [chess.square_name(s) for s in sole_defense]
        tag = f"overload:{defender_letter}{defender_name}->{','.join(asset_names)}"
        motifs.append(
            Motif(
                kind="overload",
                tag=tag,
                squares=[defender_name] + asset_names,
                description=f"{defender_letter} on {defender_name} is the sole defender of both {', '.join(asset_names)}.",
            )
        )
    return motifs


def extract_motifs(board_before: chess.Board, move: chess.Move) -> list[Motif]:
    """All tactical motifs present in the position after `move` is played
    from `board_before`, that are meaningfully tied to this move (pins and
    discoveries must be newly created; forks/skewers are read off the piece
    that just moved; hanging/trapped/overload/back-rank are read off the
    resulting position directly).
    """
    mover_color = board_before.turn
    board_after = board_before.copy(stack=False)
    board_after.push(move)

    motifs: list[Motif] = []
    motifs += _find_forks(board_after, mover_color, move.to_square)
    motifs += _find_skewers(board_after, mover_color, move.to_square)
    motifs += _find_pins(board_before, board_after, mover_color)
    motifs += _find_discoveries(board_before, board_after, mover_color, move.from_square, move.to_square)
    motifs += _find_hanging(board_after, mover_color)
    motifs += _find_back_rank(board_after, mover_color)
    motifs += _find_trapped(board_after, mover_color)
    motifs += _find_overloads(board_after, mover_color)
    return motifs
