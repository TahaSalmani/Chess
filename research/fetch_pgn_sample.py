"""Fetch a slice of a Lichess monthly database without downloading the whole 3.5 GB file.

Usage: python research/fetch_pgn_sample.py [target_games] [megabytes]

Writes artifacts/Data/raw_games.pgn, which is what stage 02 and stage 03 read.
"""
import os
import sys
import urllib.request

import zstandard as zstd

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-06.pgn.zst"
OUT_FILE = os.path.join("artifacts", "Data", "raw_games.pgn")
BYTES_PER_GAME_COMPRESSED = 200  # measured: ~775 B of PGN per game, ~4x zstd ratio


def main(target_games=6000, max_mb=None):
    if max_mb is None:
        max_mb = max(8, target_games * BYTES_PER_GAME_COMPRESSED * 1.6 // (1024 * 1024) + 4)
    chunk = max_mb * 1024 * 1024

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    req = urllib.request.Request(URL, headers={"Range": f"bytes=0-{chunk}"})
    print(f"downloading first {max_mb} MB of {URL}", flush=True)
    with urllib.request.urlopen(req) as resp:
        compressed = resp.read()
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
    count = min(target_games, len(games))
    sample = games[0] + "".join("\n\n[Event " + g for g in games[1:count])
    if not sample.endswith("\n\n"):
        sample += "\n\n"

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(sample)
    print(f"wrote {count} games ({os.path.getsize(OUT_FILE) / 1e6:.1f} MB) to {OUT_FILE}", flush=True)
    if count < target_games:
        print(f"WARNING: only {count} games in this slice, wanted {target_games}", flush=True)


if __name__ == "__main__":
    main(
        int(sys.argv[1]) if len(sys.argv) > 1 else 6000,
        int(sys.argv[2]) if len(sys.argv) > 2 else None,
    )
