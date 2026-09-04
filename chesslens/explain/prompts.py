"""The system prompt is frozen and reused verbatim across every call in a
review - same persona, same motif glossary, same grounding/output contract
- so it's the stable prefix a caching-aware model can reuse instead of
re-processing from scratch each time. Everything that varies per position
(the FEN, the engine lines, the diff, the retrieved sources) goes in the
per-call input text built by render_position_brief, never here.
"""

from __future__ import annotations

from chesslens.engine.analysis import AnalysisLine, PositionAnalysis
from chesslens.engine.classify import ClassificationResult
from chesslens.features.diff import MoveDiff
from chesslens.rag.retrieve import RetrievalResult

COACH_SYSTEM = """You are ChessLens, a chess coach explaining one move to an intermediate \
player (roughly 1200-1800 rating). Your job is not to restate the engine's evaluation - it's \
to explain, in plain language, why the recommended move is good and what the player's actual \
move gave up or gained instead.

GROUNDING CONTRACT - this is the most important rule:
You will be given a set of extracted facts about the position: engine lines, tactical motifs \
found on the board, structural pawn-formation tags, and retrieved passages from opening theory \
and instructional books. You may ONLY describe tactics, threats, and structural features that \
appear in the extracted facts you were given. Do not invent a fork, pin, or threat that isn't \
listed. Do not claim a piece is hanging unless it's tagged "hangs". If you want to reference a \
piece of chess theory or a general principle, either draw on the retrieved sources provided or \
keep it to well-known, uncontroversial general knowledge (e.g. "knights are generally worth \
about three pawns") - don't cite specific games, exact statistics, or theory you weren't given.

MOTIF VOCABULARY - these are the only tactical tags that exist. If you name a tactic, its tag \
must be in this list and must appear in motifs_cited:
fork, pin, skewer, discovered_attack, discovered_check, hangs, back_rank_weakness,
back_rank_mate_threat, trapped, overload.

AUDIENCE CALIBRATION:
Write for a player who knows the rules, basic tactics, and opening principles, but doesn't see \
deep combinations or subtle positional ideas automatically. Prefer "this leaves your knight with \
no safe squares" over "Nc3 is dominated." Name the concept the first time you use it. Keep \
centipawn talk to a minimum - translate it into what it means practically (winning, much better, \
roughly equal, worse, losing) rather than quoting raw numbers, unless the number itself is the \
point (e.g. "only a 0.2 pawn difference - essentially the same move").

OUTPUT CONTRACT:
- classification must match the classification you were given for the played move - don't \
re-derive it.
- what_your_move_misses is null when the played move WAS the engine's best move (nothing was \
missed).
- what_masters_do is null when no master-game data was provided for this position.
- motifs_cited: list only tags that were actually named in your explanation, and every one of \
them must come from the extracted facts you were given.
- moves_cited: every move you mention, in SAN, exactly as given in the position brief.
- sources: citation ids you actually drew on, from the retrieved sources list. Empty if you \
didn't use any.
- principle and next_time should generalize beyond this one position - what should the player \
watch for in future games, not just a recap of this move."""


def _format_line(analysis: PositionAnalysis, line: AnalysisLine) -> str:
    pv = " ".join(analysis.pv_san(line, limit=5))
    if line.mate is not None:
        eval_str = f"mate in {abs(line.mate)}" + (" for the opponent" if line.mate < 0 else "")
    else:
        eval_str = f"{line.cp / 100:+.2f}"
    return f"{line.san} ({eval_str}): {pv}"


def _format_motifs(motifs, label: str) -> list[str]:
    if not motifs:
        return [f"{label}: none"]
    return [f"{label}: {m.tag} - {m.description}" for m in motifs]


def _format_sources(results: list[RetrievalResult]) -> list[str]:
    lines = []
    for r in results:
        c = r.chunk
        lines.append(f'[{c.id}] ({c.source}, "{c.title}"): {c.text}')
    return lines


def render_position_brief(
    *,
    fen: str,
    mover: str,
    classification: ClassificationResult,
    analysis: PositionAnalysis,
    diff: MoveDiff,
    retrieved: list[RetrievalResult],
    master_stats_summary: str | None = None,
) -> str:
    """The per-call input: every fact the model is allowed to use, laid out
    plainly. Nothing here is prose the model should copy verbatim - it's the
    grounding it has to explain in its own words, within the vocabulary and
    contract from COACH_SYSTEM.
    """
    lines = [
        f"FEN before the move: {fen}",
        f"Side to move: {mover}",
        f"Move played: {diff.played_san}",
        f"Engine's best move: {diff.best_san}",
        f"Played move was the engine's best move: {diff.same_move}",
        f"Classification (use exactly this value): {classification.classification.value}",
        f"Win-probability lost by this move: {classification.win_percent_loss:.1f} points",
        "",
        "Top engine lines from this position:",
    ]
    for line in analysis.lines[:3]:
        lines.append(f"  {_format_line(analysis, line)}")

    lines += ["", "Tactical facts after the engine's best move:"]
    lines += ["  " + s for s in _format_motifs(diff.best_motifs, "motif")]
    lines += ["", "Tactical facts after the move actually played:"]
    lines += ["  " + s for s in _format_motifs(diff.played_motifs, "motif")]

    if diff.motifs_only_after_best:
        lines += ["", "Opportunities present after best but missing after played (the move missed these):"]
        lines += ["  " + m.tag + " - " + m.description for m in diff.motifs_only_after_best]
    if diff.motifs_only_after_played:
        lines += ["", "New problems present after played but not after best (the move created these):"]
        lines += ["  " + m.tag + " - " + m.description for m in diff.motifs_only_after_played]

    if diff.mover_material_loss_cp:
        lines.append(f"\nMaterial cost of the played move vs. best, for {mover}: {diff.mover_material_loss_cp} centipawns")
    if diff.enemy_bishop_pair_after_best != diff.enemy_bishop_pair_after_played:
        lines.append(
            f"Opponent keeps the bishop pair after played: {diff.enemy_bishop_pair_after_played} "
            f"(after best: {diff.enemy_bishop_pair_after_best})"
        )
    if diff.structure_best:
        lines.append("Pawn structure after best: " + ", ".join(s.name for s in diff.structure_best))
    if diff.structure_played:
        lines.append("Pawn structure after played: " + ", ".join(s.name for s in diff.structure_played))

    lines += ["", f"What master games show for this position: {master_stats_summary or 'no data available'}"]

    lines += ["", "Retrieved sources you may cite (id, source, title, text):"]
    if retrieved:
        lines += ["  " + s for s in _format_sources(retrieved)]
    else:
        lines.append("  (none retrieved)")

    return "\n".join(lines)
