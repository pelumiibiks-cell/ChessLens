"""ChessLens Streamlit app: explain a single position, or review a whole
game. Core logic lives in the chesslens package - this file is presentation
only, so a future FastAPI service can reuse the same calls.
"""

from __future__ import annotations

import io

import chess
import chess.pgn
import chess.svg
import streamlit as st

from chesslens.config import CORPUS_DB, get_settings
from chesslens.engine.analysis import Engine
from chesslens.engine.classify import Classification, classify_move
from chesslens.explain.client import ExplainClient, effort_for, offline_explanation
from chesslens.explain.prompts import render_position_brief
from chesslens.features.diff import diff_moves
from chesslens.rag.retrieve import retrieve
from chesslens.rag.store import CorpusStore
from chesslens.review import review_game
from chesslens.stats.select import get_stats_provider
from chesslens.verify.checks import verify_explanation

st.set_page_config(page_title="ChessLens", page_icon="♞", layout="wide")

_CLASSIFICATION_COLOR = {
    Classification.BOOK: "#6c757d",
    Classification.ONLY_MOVE: "#6c757d",
    Classification.BEST: "#2e7d32",
    Classification.GOOD: "#66bb6a",
    Classification.INACCURACY: "#f9a825",
    Classification.MISTAKE: "#ef6c00",
    Classification.BLUNDER: "#c62828",
}


def _get_store() -> CorpusStore | None:
    # Not cached: Streamlit reruns the script on a fresh thread per
    # interaction, and sqlite3 connections aren't safe to share across
    # threads. Opening a new connection per run is cheap enough not to
    # matter here.
    return CorpusStore(CORPUS_DB) if CORPUS_DB.is_file() else None


def _badge(classification: Classification) -> str:
    color = _CLASSIFICATION_COLOR.get(classification, "#6c757d")
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:10px;font-size:0.85em;">{classification.value}</span>'


def _render_board(fen: str, played_move: chess.Move | None = None, best_move: chess.Move | None = None) -> None:
    board = chess.Board(fen)
    arrows = []
    if played_move:
        arrows.append(chess.svg.Arrow(played_move.from_square, played_move.to_square, color="#c62828"))
    if best_move and best_move != played_move:
        arrows.append(chess.svg.Arrow(best_move.from_square, best_move.to_square, color="#2e7d32"))
    svg = chess.svg.board(board, arrows=arrows, size=420)
    st.components.v1.html(svg, height=440)


def _render_explanation(explanation, verification) -> None:
    st.markdown(f"**{explanation.headline}**  {_badge(explanation.classification)}", unsafe_allow_html=True)
    st.markdown(f"**Why the engine prefers this:** {explanation.why_engine_prefers}")
    if explanation.what_your_move_misses:
        st.markdown(f"**What your move misses:** {explanation.what_your_move_misses}")
    if explanation.what_masters_do:
        st.markdown(f"**What masters do here:** {explanation.what_masters_do}")
    st.markdown(f"**Principle:** {explanation.principle}")
    st.markdown(f"**Next time:** {explanation.next_time}")
    if explanation.sources:
        with st.expander(f"Sources ({len(explanation.sources)})"):
            for source_id in explanation.sources:
                st.caption(source_id)
    if not verification.ok:
        with st.expander("⚠️ Verification warnings", expanded=True):
            for err in verification.errors:
                st.warning(err)


def explain_tab() -> None:
    st.subheader("Explain a move")
    col_input, col_board = st.columns([1, 1])

    with col_input:
        fen = st.text_input("FEN", value=chess.STARTING_FEN)
        try:
            board = chess.Board(fen)
        except ValueError:
            st.error("Invalid FEN.")
            return

        move_san = st.text_input("Candidate move (SAN)", value="")
        offline = st.checkbox("Offline mode (no API calls)", value=True)
        run = st.button("Explain", type="primary")

    if not run:
        with col_board:
            _render_board(fen)
        return

    if not move_san:
        st.error("Enter a candidate move.")
        return
    try:
        move = board.parse_san(move_san)
    except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
        st.error(f"'{move_san}' is not a legal move in this position.")
        return

    settings = get_settings()
    store = _get_store()

    with st.spinner("Analysing with Stockfish..."):
        try:
            engine = Engine(settings)
        except RuntimeError as exc:
            st.error(str(exc))
            return
        with engine:
            analysis = engine.analyse(board, multipv=settings.engine_multipv)
            _cp, _mate, mover_win_after = engine.eval_after(board, move)
            classification = classify_move(
                board_before=board, analysis_before=analysis, move=move,
                mover_win_percent_after=mover_win_after, thresholds=settings.classification_thresholds,
            )
            diff = diff_moves(board, analysis.best.move, move)

    with col_board:
        _render_board(fen, played_move=move, best_move=analysis.best.move)
        st.caption(f"Red: your move ({board.san(move)})  •  Green: engine's best ({board.san(analysis.best.move)})")

    retrieved = []
    if store is not None:
        retrieved = retrieve(
            store, board, motifs=diff.best_motifs + diff.played_motifs,
            structures=diff.structure_best + diff.structure_played, k=6,
        )
        store.close()

    stats_provider = get_stats_provider(settings)
    master_stats_summary = None
    if stats_provider is not None:
        stats = stats_provider.stats_for(board)
        if stats is not None:
            master_stats_summary = stats.summary(board.turn)
        stats_provider.close()

    with st.spinner("Explaining..."):
        if offline:
            explanation = offline_explanation(classification, diff, master_stats_summary)
        else:
            client = ExplainClient(settings)
            brief = render_position_brief(
                fen=fen, mover=diff.mover, classification=classification,
                analysis=analysis, diff=diff, retrieved=retrieved,
                master_stats_summary=master_stats_summary,
            )
            try:
                explanation = client.explain(brief, classification.classification, effort=effort_for(classification.classification))
            except Exception as exc:  # noqa: BLE001 - the Gemini SDK's error hierarchy for this surface isn't publicly stable to catch narrowly
                st.error(f"Gemini API call failed: {exc}")
                return

    verification = verify_explanation(explanation, board, diff, retrieved, classification)
    st.divider()
    _render_explanation(explanation, verification)


def review_tab() -> None:
    st.subheader("Review a game")
    uploaded = st.file_uploader("PGN file", type=["pgn"])
    offline = st.checkbox("Offline mode (no API calls)", value=True, key="review_offline")
    run = st.button("Review game", type="primary")

    if not run:
        return
    if uploaded is None:
        st.error("Upload a PGN file first.")
        return

    text = uploaded.read().decode("utf-8", errors="replace")
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None:
        st.error("No game found in that file.")
        return

    settings = get_settings()
    with st.spinner("Analysing the game - this can take a while for longer games..."):
        try:
            records = review_game(game, settings=settings, offline=offline)
        except RuntimeError as exc:
            st.error(str(exc))
            return

    critical = [r for r in records if r.explanation is not None]
    st.write(f"**{len(records)}** moves analysed, **{len(critical)}** critical moments explained.")

    move_list = " ".join(
        f"{(r.ply + 1) // 2}.{'..' if r.mover == 'black' else ''}{r.san}" for r in records
    )
    st.caption(move_list)

    for record in critical:
        with st.expander(f"Ply {record.ply} ({record.mover}): {record.san} - {record.classification.classification.value}"):
            position = chess.Board(record.fen_before)
            played_move = position.parse_san(record.san)
            best_move = position.parse_san(record.diff.best_san)
            _render_board(record.fen_before, played_move=played_move, best_move=best_move)
            if record.api_error:
                st.warning(f"API call failed, showing offline template instead: {record.api_error}")
            _render_explanation(record.explanation, record.verification)


def main() -> None:
    st.title("♞ ChessLens")
    st.caption("An explainable chess coach: Stockfish + master-game statistics + RAG over chess theory.")
    tab_explain, tab_review = st.tabs(["Explain a move", "Review a game"])
    with tab_explain:
        explain_tab()
    with tab_review:
        review_tab()


if __name__ == "__main__":
    main()
