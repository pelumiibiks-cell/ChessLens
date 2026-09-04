# ChessLens

An explainable chess coach: Stockfish analysis, master-game statistics, and RAG over chess theory, explained by Gemini with a verification layer that checks the explanation against the actual position before showing it to you.

Most engine tools tell you the best move and a centipawn number. ChessLens tells you *why*: which tactical motif you missed, what structural principle applies, and what masters do in comparable positions, citing only claims it can mechanically verify against the board.

## How it works

1. **Engine analysis** — Stockfish evaluates the position and the candidate move (`chesslens/engine/`), classifying it as book, best, good, inaccuracy, mistake, or blunder based on the win-probability drop.
2. **Feature extraction** — the move is diffed against the engine's top choice: tactical motifs (forks, pins, skewers, discovered attacks, back-rank weaknesses...), static features, and pawn/piece structures (`chesslens/features/`).
3. **Master-game context** — a statistics provider (Lichess opening explorer, or a local PGN-derived table) reports how strong players actually handle this position (`chesslens/stats/`).
4. **Retrieval** — relevant chess-theory passages (openings, Wikibooks, public-domain books) are pulled from a locally built corpus via a motif- and structure-aware retriever (`chesslens/rag/`).
5. **Explanation** — Gemini generates the coaching explanation from the engine facts, motifs, and retrieved context (`chesslens/explain/`).
6. **Verification** — every claim in the explanation is checked mechanically against the position before it's shown: cited motifs must actually be present, cited moves must be legal, cited sources must be in the retrieved set, and the classification must match what the engine computed (`chesslens/verify/checks.py`). A failed check surfaces as a visible warning rather than silently passing off an unverifiable claim as fact.

An **offline mode** runs the whole pipeline except the Gemini call, producing a deterministic template explanation from the same engine facts, motifs, and stats. Useful for testing and for running without an API key.

## Install

```bash
pip install -e ".[dev]"
python scripts/setup_stockfish.py    # downloads and verifies a Stockfish binary into data/stockfish/
python scripts/build_corpus.py       # builds the RAG corpus (openings, Wikibooks, public-domain book chunks) into data/corpus.db
pytest -q                            # 65 tests
```

Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (required for the explain step outside offline mode). `LICHESS_TOKEN`, `LICHESS_USERNAME`, and `CHESSCOM_USERNAME` are optional, for the Lichess-explorer stats provider and the `fetch` command.

## Usage

**Streamlit app**, two tabs — explain a single move, or review a whole game from an uploaded PGN:

```bash
streamlit run app.py
```

**CLI**, same functionality:

```bash
chesslens explain "<FEN>" --move Nf3 [--offline]
chesslens review path/to/game.pgn [--offline] [--max N]
chesslens fetch <username> --site lichess|chesscom [--max N] [--out PATH]
```

`fetch` downloads your own recent games as a PGN file, for use as review targets.

## Tests

```bash
pytest -q
```

65 tests across engine classification, motif and structure detection, diffing, retrieval, corpus storage, stats, and the verification layer.

## License

MIT