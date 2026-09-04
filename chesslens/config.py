"""Environment loading, filesystem layout, and tunable thresholds.

No third-party .env library is used (none is in the dependency list) - the
parser below covers the KEY=VALUE / #comment / blank-line subset that
.env.example actually uses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STOCKFISH_DIR = DATA_DIR / "stockfish"
PGN_DIR = DATA_DIR / "pgn"
CORPUS_DB = DATA_DIR / "corpus.db"
STATS_DB = DATA_DIR / "stats.db"


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def load_env(env_file: Path | str | None = None) -> None:
    """Populate os.environ from a .env file, without overriding vars already set."""
    path = Path(env_file) if env_file else Path(os.environ.get("CHESSLENS_ENV_FILE", PROJECT_ROOT / ".env"))
    for key, value in _parse_env_file(path).items():
        os.environ.setdefault(key, value)


@dataclass
class Settings:
    gemini_api_key: str | None = None
    lichess_token: str | None = None
    lichess_username: str | None = None
    chesscom_username: str | None = None

    # Engine
    stockfish_path: Path | None = None
    engine_depth_ms: int = 800
    engine_multipv: int = 5

    # Explanation model
    gemini_model: str = "gemini-3.7-flash"

    # Move classification thresholds, in Lichess win-probability points (0-100).
    # A move's cost is the drop in win% for the side to move, before vs. after.
    classification_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "inaccuracy": 5.0,
            "mistake": 10.0,
            "blunder": 20.0,
        }
    )

    # Review
    max_review_positions: int = 10

    def __post_init__(self) -> None:
        if self.stockfish_path is None:
            env_path = os.environ.get("STOCKFISH_PATH")
            self.stockfish_path = Path(env_path) if env_path else _find_stockfish()


def _find_stockfish() -> Path | None:
    if not STOCKFISH_DIR.is_dir():
        return None
    for pattern in ("stockfish*.exe", "stockfish*"):
        candidates = sorted(p for p in STOCKFISH_DIR.rglob(pattern) if p.is_file())
        if candidates:
            return candidates[0]
    return None


def get_settings(env_file: Path | str | None = None) -> Settings:
    load_env(env_file)
    return Settings(
        gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
        lichess_token=os.environ.get("LICHESS_TOKEN") or None,
        lichess_username=os.environ.get("LICHESS_USERNAME") or None,
        chesscom_username=os.environ.get("CHESSCOM_USERNAME") or None,
    )
