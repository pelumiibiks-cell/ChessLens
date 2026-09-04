"""Download and verify a Stockfish binary into data/stockfish/.

Tries the AVX2 build first (fastest, works on any CPU from ~2013 onward),
falls back to the SSE4.1+POPCNT build if the AVX2 binary fails to start
(older CPU, or running under an emulator that doesn't expose AVX2).

Usage: python scripts/setup_stockfish.py
"""

from __future__ import annotations

import platform
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess.engine

from chesslens.config import STOCKFISH_DIR

RELEASE_TAG = "sf_18"
BASE_URL = f"https://github.com/official-stockfish/Stockfish/releases/download/{RELEASE_TAG}"

BUILDS = {
    "Windows": ["stockfish-windows-x86-64-avx2.zip", "stockfish-windows-x86-64-sse41-popcnt.zip"],
    "Linux": ["stockfish-ubuntu-x86-64-avx2", "stockfish-ubuntu-x86-64-sse41-popcnt"],
    "Darwin": ["stockfish-macos-x86-64-avx2", "stockfish-macos-m1-apple-silicon"],
}


def _download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "chesslens-setup"})
    with urllib.request.urlopen(request, timeout=120) as response, open(dest, "wb") as f:
        shutil.copyfileobj(response, f)


def _verify(binary: Path) -> str | None:
    """Return the engine id string on success, None on failure."""
    try:
        engine = chess.engine.SimpleEngine.popen_uci(str(binary))
        try:
            return engine.id.get("name", "unknown")
        finally:
            engine.quit()
    except Exception as exc:  # noqa: BLE001 - report and let caller try the fallback
        print(f"  handshake failed: {exc}")
        return None


def setup() -> Path:
    system = platform.system()
    if system not in BUILDS:
        raise SystemExit(f"No Stockfish build mapping for platform '{system}'. Install Stockfish manually and set STOCKFISH_PATH.")

    STOCKFISH_DIR.mkdir(parents=True, exist_ok=True)

    for asset_name in BUILDS[system]:
        print(f"trying {asset_name}")
        is_zip = asset_name.endswith(".zip")
        download_path = STOCKFISH_DIR / asset_name
        _download(f"{BASE_URL}/{asset_name}", download_path)

        if is_zip:
            with zipfile.ZipFile(download_path) as zf:
                zf.extractall(STOCKFISH_DIR)
            download_path.unlink()
            candidates = list(STOCKFISH_DIR.rglob("stockfish*.exe")) if system == "Windows" else list(STOCKFISH_DIR.rglob("stockfish*"))
        else:
            download_path.chmod(0o755)
            candidates = [download_path]

        candidates = [c for c in candidates if c.is_file()]
        if not candidates:
            print("  archive extracted but no binary found, trying next build")
            continue

        binary = candidates[0]
        name = _verify(binary)
        if name:
            print(f"OK: {name}")
            print(f"binary: {binary}")
            return binary

        print("  binary present but failed UCI handshake, trying next build")

    raise SystemExit("All Stockfish builds failed. Install manually and set STOCKFISH_PATH in .env.")


if __name__ == "__main__":
    setup()
