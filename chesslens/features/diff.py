"""Diff the position after the engine's best move against the position
after the move actually played. This is what turns "the engine prefers Nf3"
into "the engine's move keeps the bishop pair and leaves Black's queenside
pawns weak; your move trades into an ending where the extra pawn can't
promote" - the sentence is read off two feature sets, not generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

from chesslens.features.motifs import Motif, extract_motifs
from chesslens.features.static import StaticFeatures, extract_static_features
from chesslens.features.structures import PawnStructure, classify_pawn_structure


@dataclass
class MoveDiff:
    fen: str
    mover: str  # "white" | "black"
    best_san: str
    played_san: str
    same_move: bool

    best_motifs: list[Motif]
    played_motifs: list[Motif]
    motifs_only_after_best: list[Motif] = field(default_factory=list)   # opportunities the played move missed
    motifs_only_after_played: list[Motif] = field(default_factory=list)  # new problems the played move created

    best_static: StaticFeatures = None
    played_static: StaticFeatures = None

    structure_best: list[PawnStructure] = field(default_factory=list)
    structure_played: list[PawnStructure] = field(default_factory=list)

    # Positive means the played move is worse than best for the mover.
    mover_material_loss_cp: int = 0
    mover_king_safety_delta: float = 0.0  # played - best; negative = played weakens own king more
    enemy_king_safety_delta: float = 0.0  # played - best; positive = played pressures enemy king less

    # A single move can never change the MOVER's own piece count (a move
    # only ever captures the opponent's material), so "keeps/loses the
    # bishop pair" is only ever meaningful about the enemy's bishop pair -
    # i.e. whether choosing best over played wins or misses that trade.
    enemy_bishop_pair_after_best: bool = False
    enemy_bishop_pair_after_played: bool = False


def _relative_material(static: StaticFeatures, mover_key: str, enemy_key: str) -> int:
    return static.material[mover_key] - static.material[enemy_key]


def diff_moves(board_before: chess.Board, best_move: chess.Move, played_move: chess.Move) -> MoveDiff:
    mover_color = board_before.turn
    mover_key = "white" if mover_color == chess.WHITE else "black"
    enemy_key = "black" if mover_color == chess.WHITE else "white"

    best_san = board_before.san(best_move)
    played_san = board_before.san(played_move)

    board_after_best = board_before.copy(stack=False)
    board_after_best.push(best_move)
    board_after_played = board_before.copy(stack=False)
    board_after_played.push(played_move)

    best_motifs = extract_motifs(board_before, best_move)
    played_motifs = extract_motifs(board_before, played_move)
    best_tags = {m.tag for m in best_motifs}
    played_tags = {m.tag for m in played_motifs}

    best_static = extract_static_features(board_after_best)
    played_static = extract_static_features(board_after_played)

    mover_material_best = _relative_material(best_static, mover_key, enemy_key)
    mover_material_played = _relative_material(played_static, mover_key, enemy_key)

    return MoveDiff(
        fen=board_before.fen(),
        mover=mover_key,
        best_san=best_san,
        played_san=played_san,
        same_move=(best_move == played_move),
        best_motifs=best_motifs,
        played_motifs=played_motifs,
        motifs_only_after_best=[m for m in best_motifs if m.tag not in played_tags],
        motifs_only_after_played=[m for m in played_motifs if m.tag not in best_tags],
        best_static=best_static,
        played_static=played_static,
        structure_best=classify_pawn_structure(board_after_best),
        structure_played=classify_pawn_structure(board_after_played),
        mover_material_loss_cp=mover_material_best - mover_material_played,
        mover_king_safety_delta=played_static.king_safety[mover_key] - best_static.king_safety[mover_key],
        enemy_king_safety_delta=played_static.king_safety[enemy_key] - best_static.king_safety[enemy_key],
        enemy_bishop_pair_after_best=best_static.has_bishop_pair[enemy_key],
        enemy_bishop_pair_after_played=played_static.has_bishop_pair[enemy_key],
    )
