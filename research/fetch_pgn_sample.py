"""Fetch a slice of a Lichess monthly database without downloading the whole 3.5 GB file.

Usage: python research/fetch_pgn_sample.py [target_games] [megabytes]

Writes artifacts/Data/raw_games.pgn, which is what stage 02 and stage 03 read.
"""
import os
import sys
import time
import urllib.error
import urllib.request

import zstandard as zstd

URLS = [
    "https://database.lichess.org/standard/lichess_db_standard_rated_2013-06.pgn.zst",
    "https://database.lichess.org/standard/lichess_db_standard_rated_2013-07.pgn.zst",
]
OUT_FILE = os.path.join("artifacts", "Data", "raw_games.pgn")
BYTES_PER_GAME_COMPRESSED = 200  # measured: ~775 B of PGN per game, ~4x zstd ratio
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) chess-pipeline/1.0"}


def fetch_chunk(url, chunk, attempts=4):
    """Byte-range GET with retries; the endpoint throttles and occasionally 4xx/5xxes."""
    last = "no attempt made"
    for attempt in range(1, attempts + 1):
        headers = dict(HEADERS, Range=f"bytes=0-{chunk}")
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code} {e.reason}"
            print(f"  attempt {attempt}/{attempts}: {last} | body: {e.read(200)!r}", flush=True)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = repr(e)
            print(f"  attempt {attempt}/{attempts}: {last}", flush=True)
        time.sleep(5 * attempt)
    raise RuntimeError(f"giving up on {url} after {attempts} attempts: {last}")


def main(target_games=6000, max_mb=None):
    if max_mb is None:
        max_mb = max(8, target_games * BYTES_PER_GAME_COMPRESSED * 1.6 // (1024 * 1024) + 4)
    chunk = max_mb * 1024 * 1024

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    text = None
    for url in URLS:
        print(f"downloading first {max_mb} MB of {url}", flush=True)
        compressed = fetch_chunk(url, chunk)
        print(f"got {len(compressed) / 1e6:.1f} MB, decompressing", flush=True)

        reader = zstd.ZstdDecompressor().stream_reader(compressed)
        parts = []
        try:
            while True:
                block = reader.read(8 * 1024 * 1024)
                if not block:
                    break
                parts.append(block)
        except zstd.ZstdError:
            pass  # truncated stream, expected with a byte-range request
        text = b"".join(parts).decode("utf-8", errors="ignore")
        print(f"decompressed {len(text) / 1e6:.1f} MB of PGN", flush=True)

        games = text.split("\n\n[Event ")
        if len(games) >= target_games:
            break
        print(f"  only {len(games)} games in this slice, trying next mirror", flush=True)

    games = text.split("\n\n[Event ")
    count = min(target_games, len(games))
    sample = games[0] + "".join("\n\n[Event " + g for g in games[1:count])
    if not sample.endswith("\n\n"):
        sample += "\n\n"

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(sample)
    print(f"wrote {count} games ({os.path.getsize(OUT_FILE) / 1e6:.1f} MB) to {OUT_FILE}", flush=True)
    if count < target_games:
        print(f"WARNING: only {count} games in this slice, wanted {target_games}", flush=True)
    if count < max(1000, target_games // 2):
        print("ERROR: slice is far too small to train on; refusing to write a useless dataset", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main(
        int(sys.argv[1]) if len(sys.argv) > 1 else 6000,
        int(sys.argv[2]) if len(sys.argv) > 2 else None,
    )
