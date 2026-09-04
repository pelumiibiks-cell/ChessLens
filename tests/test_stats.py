import chess

from chesslens.stats.local_pgn import LocalPgnStats, build_stats_db

SAMPLE_PGN = """[Event "Test"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 1-0

[Event "Test"]
[Result "0-1"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 0-1

[Event "Test"]
[Result "1/2-1/2"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 1/2-1/2

[Event "Test"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *
"""


def test_build_and_query_stats(tmp_path):
    pgn_path = tmp_path / "test.pgn"
    pgn_path.write_text(SAMPLE_PGN)
    db_path = tmp_path / "stats.db"

    games = build_stats_db([pgn_path], db_path)
    assert games == 3  # the "*" (unclear result) game carries no signal and isn't counted

    board = chess.Board()
    for san in ["e4", "e5", "Nf3", "Nc6"]:
        board.push_san(san)

    provider = LocalPgnStats(db_path)
    stats = provider.stats_for(board)
    assert stats is not None
    assert stats.total_games == 3  # the "*" game contributed no move counts
    by_san = {m.san: m for m in stats.moves}
    assert by_san["Bb5"].white_wins == 1
    assert by_san["Bb5"].black_wins == 1
    assert by_san["Bc4"].draws == 1
    assert stats.moves[0].san == "Bb5"  # sorted by total games descending
    provider.close()


def test_unclear_result_games_are_skipped_for_stats(tmp_path):
    pgn_path = tmp_path / "test.pgn"
    pgn_path.write_text('[Event "Test"]\n[Result "*"]\n\n1. e4 e5 *\n')
    db_path = tmp_path / "stats.db"
    build_stats_db([pgn_path], db_path)

    board = chess.Board()
    provider = LocalPgnStats(db_path)
    assert provider.stats_for(board) is None
    provider.close()


def test_no_stats_db_returns_none(tmp_path):
    provider = LocalPgnStats(tmp_path / "does_not_exist.db")
    assert provider.stats_for(chess.Board()) is None
