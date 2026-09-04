"""Whole-game review: pick the handful of positions worth explaining out of
a full game, rather than paying for an explanation of all ~80 moves. That
selection is what keeps a full review to roughly ten API calls.

A position is "critical" if any of these hold:
  - the move classifies as inaccuracy or worse
  - the win-probability swing crosses a decision boundary (winning -> equal,
    equal -> losing) even if the move itself wasn't a big blunder in
    isolation
  - it's the first move that departs from the opening corpus
  - it was an only-move the player found (worth praising, not just flagging
    mistakes)
capped at settings.max_review_positions, prioritizing the worst mistakes
first if there are more critical moments than the cap allows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import chess.pgn

from chesslens.config import CORPUS_DB, Settings, get_settings
from chesslens.engine.analysis import Engine, PositionAnalysis
from chesslens.engine.classify import Classification, ClassificationResult, classify_move
from chesslens.explain.client import ExplainClient, effort_for, offline_explanation
from chesslens.explain.prompts import render_position_brief
from chesslens.explain.schema import Explanation
from chesslens.features.diff import MoveDiff, diff_moves
from chesslens.rag.retrieve import RetrievalResult, retrieve
from chesslens.rag.store import CorpusStore
from chesslens.stats.select import get_stats_provider
from chesslens.verify.checks import VerificationResult, verify_explanation

# Win% bands used to detect a decision-boundary swing, independent of
# whether any single move's loss crosses the blunder threshold.
_WINNING = 65.0
_LOSING = 35.0


@dataclass
class MoveRecord:
    ply: int
    san: str
    fen_before: str
    mover: str
    analysis: PositionAnalysis
    classification: ClassificationResult
    diff: MoveDiff
    is_critical: bool
    critical_reasons: list[str] = field(default_factory=list)
    explanation: Explanation | None = None
    verification: VerificationResult | None = None
    retrieved: list[RetrievalResult] = field(default_factory=list)
    api_error: str | None = None  # set when the live API call failed and this fell back to the offline template


def _band(win_percent: float) -> str:
    if win_percent >= _WINNING:
        return "winning"
    if win_percent <= _LOSING:
        return "losing"
    return "equal"


def analyse_game(
    game: chess.pgn.Game,
    engine: Engine,
    settings: Settings,
    store: CorpusStore | None = None,
) -> list[MoveRecord]:
    board = game.board()
    records: list[MoveRecord] = []
    off_book = False
    prev_mover_win: float | None = None

    for node in game.mainline():
        move = node.move
        mover_color = board.turn
        mover = "white" if mover_color == chess.WHITE else "black"

        analysis = engine.analyse(board, multipv=settings.engine_multipv)
        mover_cp, _mate, mover_win_after = engine.eval_after(board, move)
        result = classify_move(
            board_before=board,
            analysis_before=analysis,
            move=move,
            mover_win_percent_after=mover_win_after,
            thresholds=settings.classification_thresholds,
        )

        diff = diff_moves(board, analysis.best.move, move)

        reasons = []
        if result.classification in (Classification.INACCURACY, Classification.MISTAKE, Classification.BLUNDER):
            reasons.append(f"classified as {result.classification.value}")
        if prev_mover_win is not None:
            enemy_win_before = 100.0 - prev_mover_win
            enemy_win_after_this_move = 100.0 - mover_win_after
            if _band(enemy_win_before) != _band(enemy_win_after_this_move):
                reasons.append("crosses a winning/equal/losing boundary")
        if result.classification == Classification.ONLY_MOVE:
            reasons.append("only legal move - worth praising if found")

        if store is not None and not off_book:
            probe = board.copy(stack=False)
            probe.push(move)
            still_in_book = bool(store.by_epd(probe.epd()))
            if not still_in_book:
                reasons.append("first move outside known opening theory")
                off_book = True

        records.append(
            MoveRecord(
                ply=len(records) + 1,
                san=board.san(move),
                fen_before=board.fen(),
                mover=mover,
                analysis=analysis,
                classification=result,
                diff=diff,
                is_critical=bool(reasons),
                critical_reasons=reasons,
            )
        )

        board.push(move)
        prev_mover_win = mover_win_after

    return records


def select_critical(records: list[MoveRecord], cap: int) -> list[MoveRecord]:
    critical = [r for r in records if r.is_critical]
    critical.sort(key=lambda r: r.classification.win_percent_loss, reverse=True)
    return sorted(critical[:cap], key=lambda r: r.ply)


def explain_record(
    record: MoveRecord,
    client: ExplainClient | None,
    store: CorpusStore | None,
    offline: bool,
    stats_provider=None,
) -> None:
    """Populates record.explanation, record.retrieved, record.verification in place."""
    board = chess.Board(record.fen_before)
    retrieved: list[RetrievalResult] = []
    if store is not None:
        retrieved = retrieve(
            store,
            board,
            motifs=record.diff.best_motifs + record.diff.played_motifs,
            structures=record.diff.structure_best + record.diff.structure_played,
            k=6,
        )
    record.retrieved = retrieved

    master_stats_summary = None
    if stats_provider is not None:
        stats = stats_provider.stats_for(board)
        if stats is not None:
            master_stats_summary = stats.summary(board.turn)

    if offline or client is None:
        record.explanation = offline_explanation(record.classification, record.diff, master_stats_summary)
    else:
        brief = render_position_brief(
            fen=record.fen_before,
            mover=record.mover,
            classification=record.classification,
            analysis=record.analysis,
            diff=record.diff,
            retrieved=retrieved,
            master_stats_summary=master_stats_summary,
        )
        try:
            record.explanation = client.explain(brief, record.classification.classification, effort=effort_for(record.classification.classification))
        except Exception as exc:  # noqa: BLE001 - a failure on one move (rate limit, network blip) shouldn't discard the rest of the review
            record.api_error = str(exc)
            record.explanation = offline_explanation(record.classification, record.diff, master_stats_summary)

    record.verification = verify_explanation(record.explanation, board, record.diff, retrieved, record.classification)


def review_game(
    game: chess.pgn.Game,
    settings: Settings | None = None,
    offline: bool = False,
    store: CorpusStore | None = None,
) -> list[MoveRecord]:
    settings = settings or get_settings()
    owns_store = store is None
    if owns_store:
        store = CorpusStore(CORPUS_DB) if CORPUS_DB.is_file() else None

    client = None if offline else ExplainClient(settings)
    stats_provider = get_stats_provider(settings)

    with Engine(settings) as engine:
        records = analyse_game(game, engine, settings, store)

    critical = select_critical(records, settings.max_review_positions)
    for record in critical:
        explain_record(record, client, store, offline, stats_provider)

    if owns_store and store is not None:
        store.close()
    if stats_provider is not None:
        stats_provider.close()

    return records
