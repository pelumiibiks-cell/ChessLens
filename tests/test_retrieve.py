import chess
import pytest

from chesslens.features.motifs import Motif
from chesslens.rag.retrieve import retrieve
from chesslens.rag.schema import Chunk
from chesslens.rag.store import CorpusStore


@pytest.fixture
def ruy_lopez_board():
    board = chess.Board()
    for san in ["e4", "e5", "Nf3", "Nc6", "Bb5"]:
        board.push_san(san)
    return board


@pytest.fixture
def store(tmp_path, ruy_lopez_board):
    s = CorpusStore(tmp_path / "test.db")
    s.rebuild()
    s.add_chunks([
        Chunk(
            id="openings:C60:0001", source="openings", title="Ruy Lopez",
            text="The Ruy Lopez pins the knight on c6 against the king.",
            eco="C60", epd=ruy_lopez_board.epd(), tags=["pin"],
        ),
        Chunk(
            id="openings:C50:0001", source="openings", title="Italian Game",
            text="The Italian Game develops the bishop to c4 eyeing f7.",
            eco="C50", tags=[],
        ),
        Chunk(
            id="books:capablanca:0042", source="books", title="Chess Fundamentals",
            text="A pin along the e-file against the king is a common tactical idea.",
            tags=["pin"],
        ),
        Chunk(
            id="wikibooks:endgames:001", source="wikibooks", title="Rook Endings",
            text="In rook endings, activity matters more than material.",
            tags=[],
        ),
    ])
    s.commit()
    yield s
    s.close()


def test_ruy_lopez_position_surfaces_ruy_lopez_chunk_first(store, ruy_lopez_board):
    motifs = [Motif(kind="pin", tag="pin:Bb5->Nc6,Ke8", squares=["b5", "c6", "e8"], description="pin against the king")]
    results = retrieve(store, ruy_lopez_board, motifs=motifs, eco="C60", k=3)
    top_ids = [r.chunk.id for r in results]
    assert top_ids[0] == "openings:C60:0001"
    assert "openings:C60:0001" in top_ids[:3]


def test_exact_epd_match_gets_the_epd_channel(store, ruy_lopez_board):
    results = retrieve(store, ruy_lopez_board, k=5)
    hit = next(r for r in results if r.chunk.id == "openings:C60:0001")
    assert "epd" in hit.channels


def test_unrelated_position_does_not_surface_unrelated_opening(store):
    board = chess.Board()
    for san in ["d4", "d5", "c4", "e6"]:
        board.push_san(san)
    results = retrieve(store, board, eco="D30", k=5)
    top_ids = [r.chunk.id for r in results]
    assert "openings:C50:0001" not in top_ids[:1]


def test_empty_store_returns_no_results():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        s = CorpusStore(Path(d) / "empty.db")
        s.rebuild()
        results = retrieve(s, chess.Board(), k=5)
        assert results == []
        s.close()
