"""chesslens explain <FEN> --move <SAN> [--offline]
chesslens review <path-to.pgn> [--offline] [--max N]
chesslens fetch <username> [--site lichess|chesscom] [--max N] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys

import chess
import chess.pgn

from chesslens.config import CORPUS_DB, get_settings
from chesslens.engine.analysis import Engine
from chesslens.engine.classify import classify_move
from chesslens.explain.client import ExplainClient, effort_for, offline_explanation
from chesslens.explain.prompts import render_position_brief
from chesslens.explain.schema import Explanation
from chesslens.features.diff import diff_moves
from chesslens.rag.retrieve import retrieve
from chesslens.rag.store import CorpusStore
from chesslens.stats.select import get_stats_provider
from chesslens.verify.checks import VerificationResult, verify_explanation


def _print_explanation(explanation: Explanation, verification: VerificationResult) -> None:
    print(f"\n{explanation.headline}")
    print(f"classification: {explanation.classification.value}")
    print(f"\nwhy the engine prefers this: {explanation.why_engine_prefers}")
    if explanation.what_your_move_misses:
        print(f"\nwhat your move misses: {explanation.what_your_move_misses}")
    if explanation.what_masters_do:
        print(f"\nwhat masters do here: {explanation.what_masters_do}")
    print(f"\nprinciple: {explanation.principle}")
    print(f"next time: {explanation.next_time}")
    if explanation.sources:
        print(f"\nsources: {', '.join(explanation.sources)}")
    if not verification.ok:
        print("\n[verification warnings]")
        for err in verification.errors:
            print(f"  - {err}")


def cmd_explain(args: argparse.Namespace) -> int:
    settings = get_settings(args.env_file)
    try:
        board = chess.Board(args.fen)
    except ValueError as exc:
        print(f"'{args.fen}' is not a valid FEN: {exc}", file=sys.stderr)
        return 1
    try:
        move = board.parse_san(args.move)
    except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError) as exc:
        print(f"'{args.move}' is not a legal move in this position: {exc}", file=sys.stderr)
        return 1

    store = CorpusStore(CORPUS_DB) if CORPUS_DB.is_file() else None

    try:
        engine = Engine(settings)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    with engine:
        analysis = engine.analyse(board, multipv=settings.engine_multipv)
        _cp, _mate, mover_win_after = engine.eval_after(board, move)
        classification = classify_move(
            board_before=board,
            analysis_before=analysis,
            move=move,
            mover_win_percent_after=mover_win_after,
            thresholds=settings.classification_thresholds,
        )
        diff = diff_moves(board, analysis.best.move, move)

    retrieved = []
    if store is not None:
        retrieved = retrieve(
            store, board,
            motifs=diff.best_motifs + diff.played_motifs,
            structures=diff.structure_best + diff.structure_played,
            k=6,
        )

    stats_provider = get_stats_provider(settings)
    master_stats_summary = None
    if stats_provider is not None:
        stats = stats_provider.stats_for(board)
        if stats is not None:
            master_stats_summary = stats.summary(board.turn)
        stats_provider.close()

    if args.offline:
        explanation = offline_explanation(classification, diff, master_stats_summary)
    else:
        client = ExplainClient(settings)
        brief = render_position_brief(
            fen=args.fen, mover=diff.mover, classification=classification,
            analysis=analysis, diff=diff, retrieved=retrieved,
            master_stats_summary=master_stats_summary,
        )
        try:
            explanation = client.explain(brief, classification.classification, effort=effort_for(classification.classification))
        except Exception as exc:  # noqa: BLE001 - the Gemini SDK's error hierarchy for this surface isn't publicly stable to catch narrowly
            print(f"Gemini API call failed: {exc}", file=sys.stderr)
            if store is not None:
                store.close()
            return 1

    verification = verify_explanation(explanation, board, diff, retrieved, classification)
    _print_explanation(explanation, verification)

    if store is not None:
        store.close()
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    from chesslens.review import review_game  # deferred: pulls in ExplainClient at import time

    try:
        with open(args.pgn_path, encoding="utf-8") as f:
            game = chess.pgn.read_game(f)
    except OSError as exc:
        print(f"Couldn't read '{args.pgn_path}': {exc.strerror or exc}", file=sys.stderr)
        return 1
    if game is None:
        print(f"No game found in {args.pgn_path}", file=sys.stderr)
        return 1

    settings = get_settings(args.env_file)
    if args.max is not None:
        settings.max_review_positions = args.max

    try:
        records = review_game(game, settings=settings, offline=args.offline)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    critical = [r for r in records if r.explanation is not None]
    api_calls = 0 if args.offline else len(critical)

    print(f"moves analysed: {len(records)}")
    print(f"critical moments explained: {len(critical)}")
    print(f"API calls made: {api_calls}")

    for record in critical:
        print(f"\n{'=' * 60}")
        print(f"ply {record.ply} ({record.mover}): {record.san}  [{', '.join(record.critical_reasons)}]")
        if record.api_error:
            print(f"[API call failed, showing offline template instead: {record.api_error}]")
        _print_explanation(record.explanation, record.verification)

    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from chesslens.sources.chesscom import fetch_chesscom_games
    from chesslens.sources.lichess import fetch_lichess_games

    settings = get_settings(args.env_file)
    username = args.username or (settings.lichess_username if args.site == "lichess" else settings.chesscom_username)
    if not username:
        print(
            f"No username given, and no {'LICHESS_USERNAME' if args.site == 'lichess' else 'CHESSCOM_USERNAME'} "
            "set in .env.",
            file=sys.stderr,
        )
        return 1

    fetcher = fetch_lichess_games if args.site == "lichess" else fetch_chesscom_games
    try:
        pgn_text = fetcher(username, max_games=args.max)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:  # network failure, DNS, timeout, etc. - urllib.error.URLError is an OSError subclass
        print(f"Couldn't reach {args.site}: {exc}", file=sys.stderr)
        return 1

    if not pgn_text.strip():
        print(f"No games found for '{username}' on {args.site}.", file=sys.stderr)
        return 1

    out_path = args.out or f"{username}_{args.site}.pgn"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(pgn_text)

    game_count = pgn_text.count("[Event ")
    print(f"Saved {game_count} game(s) to {out_path}")
    print(f"Review one with: chesslens review {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chesslens", description="An explainable chess coach.")
    parser.add_argument("--env-file", dest="env_file", default=None, help="Path to a .env file (default: ./.env)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_explain = sub.add_parser("explain", help="Explain one candidate move in a position.")
    p_explain.add_argument("fen", help="FEN of the position before the move.")
    p_explain.add_argument("--move", required=True, help="Candidate move in SAN, e.g. Nf3")
    p_explain.add_argument("--offline", action="store_true", help="Use a deterministic template explanation, no API calls.")
    p_explain.set_defaults(func=cmd_explain)

    p_review = sub.add_parser("review", help="Review a full game from a PGN file.")
    p_review.add_argument("pgn_path", help="Path to a PGN file (first game only).")
    p_review.add_argument("--offline", action="store_true", help="Use template explanations, no API calls.")
    p_review.add_argument("--max", type=int, default=None, help="Max critical positions to explain (default: from settings).")
    p_review.set_defaults(func=cmd_review)

    p_fetch = sub.add_parser("fetch", help="Download a player's own recent games as a PGN file.")
    p_fetch.add_argument(
        "username", nargs="?", default=None,
        help="Lichess or Chess.com username (default: LICHESS_USERNAME/CHESSCOM_USERNAME from .env).",
    )
    p_fetch.add_argument("--site", choices=["lichess", "chesscom"], default="lichess")
    p_fetch.add_argument("--max", type=int, default=20, help="Max games to fetch (default: 20).")
    p_fetch.add_argument("--out", default=None, help="Output PGN path (default: <username>_<site>.pgn).")
    p_fetch.set_defaults(func=cmd_fetch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
