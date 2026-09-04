"""Build the RAG corpus: fetch openings, Wikibooks, and public-domain book
chunks, and load them into data/corpus.db.

Usage: python scripts/build_corpus.py [--source openings|wikibooks|books]...
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chesslens.config import CORPUS_DB
from chesslens.rag.ingest.books import fetch_book_chunks
from chesslens.rag.ingest.openings import fetch_opening_chunks
from chesslens.rag.ingest.wikibooks import fetch_wikibooks_chunks
from chesslens.rag.store import CorpusStore

SOURCES = {
    "openings": fetch_opening_chunks,
    "wikibooks": fetch_wikibooks_chunks,
    "books": fetch_book_chunks,
}


def build(sources: list[str], full_rebuild: bool) -> None:
    store = CorpusStore(CORPUS_DB)
    if full_rebuild:
        store.rebuild()

    for name in sources:
        print(f"fetching {name}...")
        t0 = time.time()
        try:
            chunks = SOURCES[name]()
        except Exception as exc:  # noqa: BLE001 - one bad source shouldn't kill the whole build
            print(f"  FAILED: {exc} (existing {name} chunks left untouched)")
            continue
        # Only clear the old chunks for this source once the fetch has
        # actually succeeded - a failed fetch (e.g. rate-limited mid-way)
        # should never wipe out data that was already there.
        if not full_rebuild:
            store.delete_by_source(name)
        n = store.add_chunks(chunks)
        store.commit()
        print(f"  {n} chunks in {time.time() - t0:.1f}s")

    print()
    print("corpus totals:")
    for source, count in sorted(store.counts().items()):
        print(f"  {source}: {count}")
    print(f"  total: {store.total()}")
    print(f"db: {CORPUS_DB}")
    store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", choices=list(SOURCES), dest="sources")
    args = parser.parse_args()
    # A full run (no --source filter) rebuilds from scratch; a filtered run
    # only touches the requested source(s) and leaves the rest of the
    # corpus alone.
    build(args.sources or list(SOURCES), full_rebuild=args.sources is None)
