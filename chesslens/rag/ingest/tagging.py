"""Shared keyword-based auto-tagging for prose sources (Wikibooks, books) -
matches the same motif/structure vocabulary features/motifs.py and
features/structures.py emit, so a chunk tagged "back_rank_mate_threat" here
lines up with the tag a position's motif extraction produces.
"""

from __future__ import annotations

import re

_TAG_KEYWORDS = {
    "fork": ["fork"],
    "pin": ["pin"],
    "skewer": ["skewer"],
    "discovered_attack": ["discovered attack"],
    "discovered_check": ["discovered check"],
    "hangs": ["hangs", "hanging piece", "en prise"],
    "back_rank_weakness": ["back rank", "back-rank"],
    "back_rank_mate_threat": ["back rank mate", "back-rank mate"],
    "trapped": ["trapped piece", "trapped bishop", "trapped knight", "trapped rook", "trapped queen"],
    "overload": ["overload", "overloaded"],
    "isolated_queen_pawn": ["isolated queen", "isolated pawn", "iqp"],
    "carlsbad": ["carlsbad"],
    "maroczy_bind": ["maroczy"],
    "hanging_pawns": ["hanging pawns"],
    "stonewall": ["stonewall"],
    "kings_indian_chain": ["king's indian", "kings indian"],
}


def _compile(keywords: list[str]) -> list[re.Pattern]:
    # \w* on a single-word keyword catches "forking"/"forks" from "fork"
    # while still requiring a word boundary, so "pin" doesn't match inside
    # "opinion" - a real false positive naive substring matching produced.
    patterns = []
    for kw in keywords:
        if " " in kw or "-" in kw:
            patterns.append(re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE))
        else:
            patterns.append(re.compile(r"\b" + re.escape(kw) + r"\w*\b", re.IGNORECASE))
    return patterns


_COMPILED_KEYWORDS = {tag: _compile(keywords) for tag, keywords in _TAG_KEYWORDS.items()}


def auto_tags(text: str) -> list[str]:
    return [tag for tag, patterns in _COMPILED_KEYWORDS.items() if any(p.search(text) for p in patterns)]
