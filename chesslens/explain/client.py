"""Gemini client for the explanation step, via the Interactions API
(client.interactions.create - google-genai >= 2.3.0). Structured output is
enforced through response_format so the result parses directly into
Explanation; thinking_level is tiered by how much is riding on the move,
so routine book moves don't spend the same reasoning budget as a blunder.

Includes an offline path (`offline_explanation`) that builds an Explanation
straight from the extracted facts with no model call at all - every other
stage of the pipeline can be developed and tested against it without
spending anything, and it's also the fallback the CLI/review loop can use
if the API key isn't configured.
"""

from __future__ import annotations

from google import genai

from chesslens.config import Settings, get_settings
from chesslens.engine.classify import Classification, ClassificationResult
from chesslens.explain.prompts import COACH_SYSTEM
from chesslens.explain.schema import Explanation
from chesslens.features.diff import MoveDiff

_EFFORT_BY_CLASSIFICATION = {
    Classification.BLUNDER: "high",
    Classification.MISTAKE: "high",
    Classification.INACCURACY: "medium",
    Classification.GOOD: "medium",
    Classification.BEST: "low",
    Classification.BOOK: "low",
    Classification.ONLY_MOVE: "low",
}


def effort_for(classification: Classification) -> str:
    return _EFFORT_BY_CLASSIFICATION.get(classification, "medium")


class ExplainClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client: genai.Client | None = None

    def _client_or_raise(self) -> genai.Client:
        if self._client is None:
            if not self.settings.gemini_api_key:
                raise RuntimeError(
                    "No GEMINI_API_KEY configured. Set it in .env, or pass --offline to use "
                    "template explanations with no API calls."
                )
            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    def explain(self, brief: str, classification: Classification, effort: str | None = None) -> Explanation:
        client = self._client_or_raise()
        interaction = client.interactions.create(
            model=self.settings.gemini_model,
            system_instruction=COACH_SYSTEM,
            input=brief,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": Explanation.model_json_schema(),
            },
            generation_config={"thinking_level": effort or effort_for(classification)},
        )
        if interaction.status != "completed":
            error_detail = "; ".join(str(e) for e in (interaction.errors or []))
            raise RuntimeError(f"Gemini interaction did not complete (status={interaction.status}): {error_detail}")
        if not interaction.output_text:
            raise RuntimeError("Gemini interaction completed with no output_text.")
        return Explanation.model_validate_json(interaction.output_text)


def offline_explanation(classification: ClassificationResult, diff: MoveDiff, master_stats_summary: str | None = None) -> Explanation:
    """Deterministic, no-API explanation built straight from extracted
    facts - not eloquent, but every field is grounded and verify.py-clean
    by construction, which makes it useful for testing the rest of the
    pipeline without spending anything. master_stats_summary is local data
    (not an API call), so "offline" doesn't need to mean "without it".
    """
    c = classification.classification
    motifs_missed = diff.motifs_only_after_best
    motifs_created = diff.motifs_only_after_played

    if c in (Classification.BEST, Classification.BOOK, Classification.ONLY_MOVE):
        headline = f"{diff.played_san} was the engine's top choice."
        misses = None
    else:
        headline = f"{diff.played_san} cost about {classification.win_percent_loss:.0f} win-probability points compared to {diff.best_san}."
        parts = []
        if motifs_missed:
            parts.append("it misses " + "; ".join(m.description for m in motifs_missed))
        if diff.mover_material_loss_cp > 0:
            parts.append(f"it gives up roughly {diff.mover_material_loss_cp} centipawns of material compared to best")
        if motifs_created:
            parts.append("it creates a new problem: " + "; ".join(m.description for m in motifs_created))
        if parts:
            cleaned = [p.rstrip(".") for p in parts]
            misses = ". ".join(p[0].upper() + p[1:] for p in cleaned) + "."
        else:
            misses = f"{diff.best_san} was simply more accurate here."

    why_best = f"{diff.best_san} is the engine's top choice in this position."
    if diff.best_motifs:
        why_best += " It also " + "; ".join(m.description for m in diff.best_motifs)

    motifs_cited = sorted({m.kind for m in (diff.best_motifs + diff.played_motifs)})
    moves_cited = sorted({diff.best_san, diff.played_san})

    return Explanation(
        headline=headline,
        classification=c,
        why_engine_prefers=why_best,
        what_your_move_misses=misses,
        what_masters_do=master_stats_summary,
        principle="Compare what each candidate move keeps or gives up, not just the resulting evaluation number.",
        next_time="Before moving, check whether your move leaves any piece attacked or gives up a tactical resource the alternative kept.",
        motifs_cited=motifs_cited,
        moves_cited=moves_cited,
        sources=[],
    )
