"""Download a modest set of elite tournament PGNs (pgnmentor.com, 2010+,
~2.5MB / 49 events) into data/pgn/, then build data/stats.db from them.

Usage: python scripts/build_stats.py
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chesslens.config import PGN_DIR, STATS_DB
from chesslens.stats.local_pgn import build_stats_db

BASE_URL = "https://www.pgnmentor.com"
USER_AGENT = "chesslens-corpus-builder"

# Recent elite round-robins and knockouts - enough games to have real
# opening statistics without pulling gigabytes. See scripts/build_corpus.py
# for the theory/prose corpus; this is the separate empirical channel.
WANTED_EVENT_PREFIXES = ["WijkaanZee", "Stavanger", "SaintLouis", "Candidates", "Bucharest"]
MIN_YEAR = 2010


def _list_event_pgns() -> list[str]:
    request = urllib.request.Request(f"{BASE_URL}/files.html", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
    links = sorted(set(re.findall(r'href="(events/[^"]+\.pgn)"', html)))

    picked = []
    for link in links:
        name = link.split("/")[1]
        if not any(name.startswith(p) for p in WANTED_EVENT_PREFIXES):
            continue
        year_match = re.search(r"(\d{4})", name)
        if year_match and int(year_match.group(1)) >= MIN_YEAR:
            picked.append(link)
    return picked


def _download(link: str, dest_dir: Path) -> Path:
    dest = dest_dir / Path(link).name
    if not dest.is_file():
        request = urllib.request.Request(f"{BASE_URL}/{link}", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response, open(dest, "wb") as f:
            f.write(response.read())
    return dest


def main() -> None:
    PGN_DIR.mkdir(parents=True, exist_ok=True)
    links = _list_event_pgns()
    print(f"found {len(links)} events, downloading into {PGN_DIR}...")

    paths = []
    for link in links:
        try:
            paths.append(_download(link, PGN_DIR))
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't kill the build
            print(f"  FAILED {link}: {exc}")

    print(f"downloaded {len(paths)} files, building {STATS_DB}...")
    games = build_stats_db(paths, STATS_DB)
    print(f"ingested {games} games into position_stats")


if __name__ == "__main__":
    main()
