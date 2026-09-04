import chess

from chesslens.features.static import extract_static_features


def test_start_position_is_balanced_opening():
    f = extract_static_features(chess.Board())
    assert f.material_diff == 0
    assert f.phase == "opening"
    assert f.mobility == {"white": 20, "black": 20}
    assert f.has_bishop_pair == {"white": True, "black": True}
    assert f.outposts == []


def test_isolated_pawn_detected():
    f = extract_static_features(chess.Board("4k3/pp3ppp/8/8/3P4/8/PP3PPP/4K3 w - - 0 1"))
    assert f.pawns["white"].isolated == ["d4"]


def test_doubled_pawns_detected():
    f = extract_static_features(chess.Board("4k3/8/8/8/4P3/4P3/8/4K3 w - - 0 1"))
    assert sorted(f.pawns["white"].doubled) == ["e3", "e4"]


def test_passed_pawn_detected():
    f = extract_static_features(chess.Board("4k3/8/3P4/8/8/8/8/4K3 w - - 0 1"))
    assert f.pawns["white"].passed == ["d6"]


def test_backward_pawn_detected():
    # White c2 pawn, no b/d file support, and its advance square c3 is
    # covered by black pawns on b4 and d4.
    f = extract_static_features(chess.Board("4k3/8/8/8/1p1p4/8/2P5/4K3 w - - 0 1"))
    assert f.pawns["white"].backward == ["c2"]


def test_outpost_detected():
    # Knight on d5, supported by the e4 pawn, no black pawns on c/e files
    # that could ever challenge it.
    f = extract_static_features(chess.Board("4k3/8/8/3N4/4P3/8/8/4K3 w - - 0 1"))
    assert f.outposts == ["d5"]


def test_king_safety_penalizes_open_files_and_rewards_shield():
    exposed = extract_static_features(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"))
    shielded = extract_static_features(chess.Board("4k3/8/8/8/8/8/PPP5/2K5 w - - 0 1"))
    assert shielded.king_safety["white"] > exposed.king_safety["white"]
