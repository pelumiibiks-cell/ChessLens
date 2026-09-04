import chess

from chesslens.features.diff import diff_moves


def board_from(pieces: dict[str, str], turn: chess.Color = chess.WHITE) -> chess.Board:
    board = chess.Board.empty()
    for square_name, symbol in pieces.items():
        color = chess.WHITE if symbol.isupper() else chess.BLACK
        piece_type = chess.Piece.from_symbol(symbol).piece_type
        board.set_piece_at(chess.parse_square(square_name), chess.Piece(piece_type, color))
    board.turn = turn
    return board


def test_material_loss_from_missing_a_free_capture():
    board = board_from({"e1": "K", "d4": "P", "e8": "k", "e5": "q"})
    diff = diff_moves(board, board.parse_san("dxe5"), board.parse_san("Kd1"))
    assert diff.mover_material_loss_cp == 900
    assert diff.best_san == "dxe5"
    assert diff.played_san == "Kd1"
    assert not diff.same_move


def test_same_move_flag():
    board = board_from({"e1": "K", "d4": "P", "e8": "k", "e5": "q"})
    move = board.parse_san("dxe5")
    diff = diff_moves(board, move, move)
    assert diff.same_move
    assert diff.mover_material_loss_cp == 0


def test_enemy_bishop_pair_won_by_best_move_but_missed_by_played():
    # White knight on h5 can capture the bishop on f6, breaking Black's
    # bishop pair; a quiet knight retreat does not.
    board = board_from({"e1": "K", "h5": "N", "g1": "N", "e8": "k", "f6": "b", "c8": "b"})
    diff = diff_moves(board, board.parse_san("Nxf6"), board.parse_san("Nh3"))
    assert diff.enemy_bishop_pair_after_best is False
    assert diff.enemy_bishop_pair_after_played is True


def test_motif_diff_isolates_the_missed_fork():
    board = board_from({"e1": "K", "c3": "N", "e8": "k", "d6": "q", "f6": "r"})
    diff = diff_moves(board, board.parse_san("Ne4"), board.parse_san("Ke2"))
    best_only_tags = {m.tag for m in diff.motifs_only_after_best}
    played_only_tags = {m.tag for m in diff.motifs_only_after_played}
    assert "fork:Ne4->d6,f6" in best_only_tags
    assert played_only_tags == set()
