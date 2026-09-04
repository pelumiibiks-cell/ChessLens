import chess
import pytest

from chesslens.rag.schema import Chunk
from chesslens.rag.store import CorpusStore


@pytest.fixture
def store(tmp_path):
    s = CorpusStore(tmp_path / "test.db")
    s.rebuild()
    yield s
    s.close()


def test_add_and_get_roundtrip(store):
    chunk = Chunk(id="x:1", source="books", title="T", text="hello world", tags=["pin"])
    store.add_chunk(chunk)
    store.commit()
    fetched = store.get("x:1")
    assert fetched.title == "T"
    assert fetched.tags == ["pin"]


def test_by_epd_exact_match_only(store):
    board = chess.Board()
    board.push_san("e4")
    epd = board.epd()
    store.add_chunks([
        Chunk(id="a", source="openings", title="A", text="king pawn opening", epd=epd),
        Chunk(id="b", source="openings", title="B", text="queen pawn opening", epd="different-epd"),
    ])
    store.commit()
    results = store.by_epd(epd)
    assert [c.id for c in results] == ["a"]


def test_by_eco_prefix_matches_family(store):
    store.add_chunks([
        Chunk(id="a", source="openings", title="A", text="x", eco="B12"),
        Chunk(id="b", source="openings", title="B", text="y", eco="B10"),
        Chunk(id="c", source="openings", title="C", text="z", eco="C60"),
    ])
    store.commit()
    results = {c.id for c in store.by_eco_prefix("B1")}
    assert results == {"a", "b"}


def test_by_tags_ranks_more_overlap_higher(store):
    store.add_chunks([
        Chunk(id="a", source="books", title="A", text="x", tags=["pin", "fork"]),
        Chunk(id="b", source="books", title="B", text="y", tags=["pin"]),
        Chunk(id="c", source="books", title="C", text="z", tags=["skewer"]),
    ])
    store.commit()
    results = store.by_tags(["pin", "fork"])
    assert [c.id for c in results] == ["a", "b"]


def test_bm25_search_finds_relevant_text(store):
    store.add_chunks([
        Chunk(id="a", source="books", title="A", text="The bishop pins the knight against the king."),
        Chunk(id="b", source="books", title="B", text="The rook controls the open file."),
    ])
    store.commit()
    results = store.bm25_search("pin knight")
    assert results[0][0].id == "a"


def test_bm25_search_ignores_stopwords_only_query(store):
    store.add_chunks([Chunk(id="a", source="books", title="A", text="hello")])
    store.commit()
    assert store.bm25_search("the a an") == []
