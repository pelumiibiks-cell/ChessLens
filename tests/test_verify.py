import chess

from chesslens.engine.classify import Classification, ClassificationResult
from chesslens.explain.schema import Explanation
from chesslens.features.diff import diff_moves
from chesslens.verify.checks import verify_explanation


def board_from(pieces: dict[str, str], turn: chess.Color = chess.WHITE) -> chess.Board:
    board = chess.Board.empty()
    for square_name, symbol in pieces.items():
        color = chess.WHITE if symbol.isupper() else chess.BLACK
        piece_type = chess.Piece.from_symbol(symbol).piece_type
        board.set_piece_at(chess.parse_square(square_name), chess.Piece(piece_type, color))
    board.turn = turn
    return board


def make_diff_and_result():
    board = board_from({"e1": "K", "d4": "P", "e8": "k", "e5": "q"})
    best = board.parse_san("dxe5")
    played = board.parse_san("Kd1")
    diff = diff_moves(board, best, played)
    result = ClassificationResult(Classification.BLUNDER, 37.0)
    return board, diff, result


def valid_explanation(diff) -> Explanation:
    return Explanation(
        headline="Kd1 hangs the free queen capture.",
        classification=Classification.BLUNDER,
        why_engine_prefers="dxe5 wins the queen for free.",
        what_your_move_misses="It misses the free capture on e5, and the pawn on d4 is also left hanging.",
        what_masters_do=None,
        principle="Always check for free captures before making a quiet move.",
        next_time="Scan for undefended enemy pieces before playing.",
        motifs_cited=["hangs"],
        moves_cited=["Kd1", "dxe5"],
        sources=[],
    )


def test_valid_explanation_passes():
    board, diff, result = make_diff_and_result()
    explanation = valid_explanation(diff)
    v = verify_explanation(explanation, board, diff, retrieved=[], classification=result)
    assert v.ok
    assert v.errors == []


def test_rejects_ungrounded_motif():
    board, diff, result = make_diff_and_result()
    explanation = valid_explanation(diff)
    explanation.motifs_cited = ["fork"]  # not actually present in this position
    v = verify_explanation(explanation, board, diff, retrieved=[], classification=result)
    assert not v.ok
    assert any("fork" in e for e in v.errors)


def test_accepts_move_only_legal_one_ply_later_as_a_threat():
    # Real case found in production: white's knight already attacks the e5
    # pawn; it's black to move. The model correctly warned "...and now
    # White plays Nxe5 next" - but "Nxe5" is a WHITE move, so it's not
    # parseable in board_before at all (black to move there), only in the
    # position one ply later, after black's actual move. That's legitimate
    # coaching language describing a threat, not a hallucination.
    board = board_from({"e1": "K", "e8": "k", "f3": "N", "e5": "p", "a7": "p"}, turn=chess.BLACK)
    assert not board.is_legal(chess.Move.from_uci("f3e5"))  # sanity: it's not White's move here
    best = board.parse_san("a6")  # doesn't address the threat
    played = board.parse_san("a6")
    diff = diff_moves(board, best, played)
    result = ClassificationResult(Classification.BEST, 0.0)
    explanation = valid_explanation(diff)
    explanation.classification = Classification.BEST
    explanation.what_your_move_misses = None
    explanation.moves_cited = ["a6", "Nxe5"]
    v = verify_explanation(explanation, board, diff, retrieved=[], classification=result)
    assert v.ok, v.errors


def test_rejects_motif_outside_vocabulary():
    board, diff, result = make_diff_and_result()
    explanation = valid_explanation(diff)
    explanation.motifs_cited = ["windmill"]  # not a real tag at all
    v = verify_explanation(explanation, board, diff, retrieved=[], classification=result)
    assert not v.ok
    assert any("vocabulary" in e for e in v.errors)


def test_rejects_illegal_move_citation():
    board, diff, result = make_diff_and_result()
    explanation = valid_explanation(diff)
    explanation.moves_cited = ["Qxd1"]  # no black queen move can reach d1 here
    v = verify_explanation(explanation, board, diff, retrieved=[], classification=result)
    assert not v.ok
    assert any("Qxd1" in e for e in v.errors)


def test_rejects_unknown_source_citation():
    board, diff, result = make_diff_and_result()
    explanation = valid_explanation(diff)
    explanation.sources = ["books:made_up:9999"]
    v = verify_explanation(explanation, board, diff, retrieved=[], classification=result)
    assert not v.ok
    assert any("made_up" in e for e in v.errors)


def test_rejects_mismatched_classification():
    board, diff, result = make_diff_and_result()
    explanation = valid_explanation(diff)
    explanation.classification = Classification.BEST
    v = verify_explanation(explanation, board, diff, retrieved=[], classification=result)
    assert not v.ok
    assert any("classification" in e for e in v.errors)


def test_accepts_null_misses_when_move_is_best_by_tolerance_not_identity():
    # classify_move labels a move BEST whenever it's within tolerance of the
    # top engine choice, not only when it's literally the same move object -
    # so diff.same_move can be False even though nothing was missed. The
    # null-misses check must key off classification, not move identity.
    board = board_from({"e1": "K", "e8": "k"})
    best = board.parse_san("Kd1")
    played = board.parse_san("Kf1")  # a different, equally fine king move
    diff = diff_moves(board, best, played)
    assert not diff.same_move
    result = ClassificationResult(Classification.BEST, 0.1)  # within tolerance of best
    explanation = valid_explanation(diff)
    explanation.classification = Classification.BEST
    explanation.what_your_move_misses = None
    explanation.motifs_cited = []
    explanation.moves_cited = ["Kf1"]
    v = verify_explanation(explanation, board, diff, retrieved=[], classification=result)
    assert v.ok, v.errors


def test_rejects_misses_field_present_when_move_was_best():
    board = board_from({"e1": "K", "d4": "P", "e8": "k", "e5": "q"})
    best = board.parse_san("dxe5")
    diff = diff_moves(board, best, best)  # played == best
    result = ClassificationResult(Classification.BEST, 0.0)
    explanation = valid_explanation(diff)
    explanation.classification = Classification.BEST
    explanation.what_your_move_misses = "some spurious claim"
    v = verify_explanation(explanation, board, diff, retrieved=[], classification=result)
    assert not v.ok
    assert any("what_your_move_misses" in e for e in v.errors)
