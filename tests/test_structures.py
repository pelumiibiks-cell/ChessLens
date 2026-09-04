import chess

from chesslens.features.structures import classify_pawn_structure


def board_from(pieces: dict[str, str]) -> chess.Board:
    board = chess.Board.empty()
    for square_name, symbol in pieces.items():
        color = chess.WHITE if symbol.isupper() else chess.BLACK
        piece_type = chess.Piece.from_symbol(symbol).piece_type
        board.set_piece_at(chess.parse_square(square_name), chess.Piece(piece_type, color))
    return board


def names(board: chess.Board) -> set[str]:
    return {s.name for s in classify_pawn_structure(board)}


def test_isolated_queens_pawn():
    board = board_from({
        "d4": "P", "a2": "P", "b2": "P", "f2": "P", "g2": "P", "h2": "P",
        "a7": "p", "b7": "p", "c7": "p", "e6": "p", "f7": "p", "g7": "p", "h7": "p",
    })
    assert names(board) == {"isolated_queen_pawn"}


def test_carlsbad():
    board = board_from({
        "a2": "P", "b2": "P", "d4": "P", "f2": "P", "g2": "P", "h2": "P",
        "a7": "p", "b7": "p", "d5": "p", "f7": "p", "g7": "p", "h7": "p",
    })
    assert "carlsbad" in names(board)


def test_maroczy_bind():
    board = board_from({"c4": "P", "e4": "P", "a2": "P", "b2": "P", "f2": "P", "g2": "P", "h2": "P"})
    assert names(board) == {"maroczy_bind"}


def test_hanging_pawns_advanced():
    board = board_from({"c4": "P", "d4": "P", "a2": "P", "f2": "P", "g2": "P", "h2": "P"})
    assert names(board) == {"hanging_pawns"}


def test_home_rank_pawns_are_not_hanging_pawns():
    # a2/b2 sit on their start squares with an empty c-file - structurally
    # identical to the "hanging pawns" flank check but not a real weakness.
    board = board_from({"a2": "P", "b2": "P", "d4": "P", "f2": "P", "g2": "P", "h2": "P"})
    assert "hanging_pawns" not in names(board)


def test_stonewall():
    board = board_from({"d4": "P", "e3": "P", "f4": "P", "a2": "P", "b2": "P", "c2": "P", "g2": "P", "h2": "P"})
    assert "stonewall" in names(board)


def test_kings_indian_chain():
    board = board_from({"d4": "P", "e4": "P", "d6": "p", "e5": "p", "a2": "P", "b2": "P"})
    assert "kings_indian_chain" in names(board)


def test_open_position_has_no_structure_tags():
    board = board_from({"a2": "P", "h2": "P", "a7": "p", "h7": "p"})
    assert classify_pawn_structure(board) == []
