import chess

from chesslens.features.see import hanging_value, static_exchange_eval


def test_undefended_pawn_capture():
    board = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    move = board.parse_san("exd5")
    assert static_exchange_eval(board, move) == 100


def test_defended_pawn_capture_is_even_trade():
    board = chess.Board("4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1")
    move = board.parse_san("exd5")
    assert static_exchange_eval(board, move) == 0


def test_bad_trade_knight_for_defended_pawn():
    board = chess.Board("4k3/8/1p6/2p5/8/1N6/8/4K3 w - - 0 1")
    move = board.parse_san("Nxc5")
    assert static_exchange_eval(board, move) == 100 - 320


def test_rook_takes_undefended_pawn():
    board = chess.Board("4k3/8/8/8/8/8/3p4/3RK3 w - - 0 1")
    move = board.parse_san("Rxd2")
    assert static_exchange_eval(board, move) == 100


def test_xray_battery_recapture():
    # Two white rooks stacked on the d-file behind each other; black has a
    # single rook defending the d5 pawn. White should net pawn + rook - rook.
    board = chess.Board("1k1r4/8/8/3p4/8/8/3R4/1K1R4 w - - 0 1")
    move = board.parse_san("Rxd5")
    assert static_exchange_eval(board, move) == 100


def test_noncapture_move_has_zero_see():
    board = chess.Board()
    move = board.parse_san("e4")
    assert static_exchange_eval(board, move) == 0


def test_hanging_value_reports_queen_free_for_the_taking():
    board = chess.Board("4k3/8/8/4q3/3P4/8/8/4K3 w - - 0 1")
    assert hanging_value(board, chess.E5) == 900


def test_hanging_value_none_when_not_attacked():
    board = chess.Board()
    assert hanging_value(board, chess.E7) is None
