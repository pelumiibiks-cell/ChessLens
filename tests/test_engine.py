import chess
import pytest

from chesslens.config import get_settings
from chesslens.engine import Classification, Engine, classify_move, cp_to_win_percent

FOOLS_MATE_FEN = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
ONLY_MOVE_FEN = "k7/8/1K6/8/8/8/8/7Q b - - 0 1"


@pytest.fixture(scope="module")
def engine():
    settings = get_settings()
    with Engine(settings) as eng:
        yield eng


def test_win_percent_midpoint_is_even():
    assert cp_to_win_percent(0) == pytest.approx(50.0)


def test_win_percent_monotonic():
    assert cp_to_win_percent(200) > cp_to_win_percent(0) > cp_to_win_percent(-200)


def test_finds_mate_in_one(engine):
    board = chess.Board(FOOLS_MATE_FEN)
    analysis = engine.analyse(board, multipv=1, movetime_ms=300)
    assert analysis.best.mate == 1
    assert board.san(analysis.best.move) == "Qh4#"
    assert analysis.best.win_percent == pytest.approx(100.0, abs=1e-6)


def test_only_move_classified(engine):
    board = chess.Board(ONLY_MOVE_FEN)
    assert board.legal_moves.count() == 1
    analysis = engine.analyse(board, multipv=1, movetime_ms=200)
    move = analysis.best.move
    settings = get_settings()
    mover_cp, _mate, mover_win = engine.eval_after(board, move, movetime_ms=200)
    result = classify_move(
        board_before=board,
        analysis_before=analysis,
        move=move,
        mover_win_percent_after=mover_win,
        thresholds=settings.classification_thresholds,
    )
    assert result.classification == Classification.ONLY_MOVE


def test_declining_free_queen_capture_is_a_blunder(engine):
    # Black queen hangs to dxe5 with nothing else contested on the board.
    # Capturing it should be "best"; any quiet non-capture lets the queen
    # escape next move and must classify as a blunder - a large, unambiguous
    # win%-loss regardless of exact engine depth.
    board = chess.Board("4k3/8/8/4q3/3P4/8/8/4K3 w - - 0 1")
    analysis = engine.analyse(board, multipv=1, movetime_ms=300)
    settings = get_settings()

    capture = board.parse_san("dxe5")
    assert analysis.best.move == capture

    quiet_move = board.parse_san("Kd1")
    mover_cp, _mate, mover_win = engine.eval_after(board, quiet_move, movetime_ms=300)
    result = classify_move(
        board_before=board,
        analysis_before=analysis,
        move=quiet_move,
        mover_win_percent_after=mover_win,
        thresholds=settings.classification_thresholds,
    )
    assert result.classification == Classification.BLUNDER
    assert result.win_percent_loss >= settings.classification_thresholds["blunder"]
