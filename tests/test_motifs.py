import chess
import pytest

from chesslens.features.motifs import extract_motifs


def board_from(pieces: dict[str, str], turn: chess.Color = chess.WHITE) -> chess.Board:
    board = chess.Board.empty()
    for square_name, symbol in pieces.items():
        color = chess.WHITE if symbol.isupper() else chess.BLACK
        piece_type = chess.Piece.from_symbol(symbol).piece_type
        board.set_piece_at(chess.parse_square(square_name), chess.Piece(piece_type, color))
    board.turn = turn
    return board


def kinds(board: chess.Board, move_san: str) -> set[str]:
    move = board.parse_san(move_san)
    return {m.kind for m in extract_motifs(board, move)}


def tags(board: chess.Board, move_san: str) -> set[str]:
    move = board.parse_san(move_san)
    return {m.tag for m in extract_motifs(board, move)}


def test_fork():
    board = board_from({"e1": "K", "c3": "N", "e8": "k", "d6": "q", "f6": "r"})
    t = tags(board, "Ne4")
    assert "fork:Ne4->d6,f6" in t


def test_pin():
    board = board_from({"e1": "K", "c4": "B", "e8": "k", "d7": "n"})
    t = tags(board, "Bb5")
    assert "pin:Bb5->Nd7,Ke8" in t


def test_skewer():
    board = board_from({"e1": "K", "d1": "R", "e8": "k", "d7": "q", "d8": "r"})
    t = tags(board, "Rd5")
    assert "skewer:Rd5->Qd7,Rd8" in t


def test_discovered_check():
    board = board_from({"e1": "K", "d1": "R", "d5": "N", "d8": "k"})
    t = tags(board, "Nc7+")
    assert "discovered_check:Rd1->Kd8" in t


def test_hangs_own_piece():
    board = board_from({"e1": "K", "d5": "R", "e8": "k", "c6": "b"})
    t = tags(board, "Kd2")
    assert "hangs:Rd5" in t


def test_back_rank_mate_threat():
    board = board_from({"e1": "K", "a1": "R", "g8": "k", "f7": "p", "g7": "p", "h7": "p"})
    t = tags(board, "Ra8+")
    assert "back_rank_mate_threat:Kg8" in t


def test_trapped_piece():
    # Knight in the corner; both its escape squares are covered by bishops,
    # and the king is kept far away so it doesn't also defend them.
    board = board_from({"e1": "K", "a5": "k", "h8": "n", "c4": "B", "b1": "B"})
    t = tags(board, "Kf1")
    assert "trapped:Nh8" in t


def test_overload():
    # Knight is the sole defender of both an attacked rook and an attacked
    # pawn; king kept away so it isn't a second defender of either.
    board = board_from({"e1": "K", "a8": "k", "e5": "n", "d7": "r", "g6": "p", "b7": "R", "c2": "B"})
    t = tags(board, "Kf1")
    assert "overload:Ne5->g6,d7" in t


def test_quiet_move_with_no_tactics_yields_no_motifs():
    board = chess.Board()
    assert extract_motifs(board, board.parse_san("Nf3")) == []


@pytest.mark.parametrize(
    "fen,move_san",
    [
        ("4k3/8/3q1r2/8/8/2N5/8/4K3 w - - 0 1", "Ne4"),
        ("4k3/3n4/8/8/2B5/8/8/4K3 w - - 0 1", "Bb5"),
    ],
)
def test_motif_squares_are_legal_on_the_board(fen, move_san):
    board = chess.Board(fen)
    move = board.parse_san(move_san)
    for motif in extract_motifs(board, move):
        for square_name in motif.squares:
            chess.parse_square(square_name)  # raises if malformed
