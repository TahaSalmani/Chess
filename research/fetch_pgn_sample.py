"""Fetch a slice of Lichess monthly databases without downloading the whole 3.5 GB files.

Usage: python research/fetch_pgn_sample.py [target_games] [megabytes_per_month]

Samples target_games spread over several months (via HTTP byte-range requests of a few
MB each) so the opening/style diversity is wider than one slice of one month.
Writes artifacts/Data/raw_games.pgn, which is what stage 02 and stage 03 read.
"""
import os
import sys
import time
import urllib.error
import urllib.request

import zstandard as zstd

URLS = [
    "https://database.lichess.org/standard/lichess_db_standard_rated_2014-06.pgn.zst",
    "https://database.lichess.org/standard/lichess_db_standard_rated_2016-06.pgn.zst",
    "https://database.lichess.org/standard/lichess_db_standard_rated_2018-06.pgn.zst",
    "https://database.lichess.org/standard/lichess_db_standard_rated_2020-06.pgn.zst",
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


def decompress(compressed):
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
    return b"".join(parts).decode("utf-8", errors="ignore")


def fetch_month(url, want_games, max_mb):
    chunk = max_mb * 1024 * 1024
    print(f"downloading first {max_mb} MB of {url}", flush=True)
    text = decompress(fetch_chunk(url, chunk))
    games = text.split("\n\n[Event ")
    count = min(want_games, len(games))
    sample = games[0] + "".join("\n\n[Event " + g for g in games[1:count])
    print(f"  got {count} games from this slice (slice held {len(games)})", flush=True)
    return sample, count


def main(target_games=25000, max_mb=None):
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    parts = []
    total = 0
    for idx, url in enumerate(URLS):
        remaining_months = len(URLS) - idx
        want = -((target_games - total) // -remaining_months)  # spread evenly, top up later
        if want <= 0:
            break
        mb = max_mb if max_mb else max(8, want * BYTES_PER_GAME_COMPRESSED * 1.6 // (1024 * 1024) + 4)
        try:
            sample, count = fetch_month(url, want, mb)
        except RuntimeError as e:
            print(f"  skipping {url}: {e}", flush=True)
            continue
        if count:
            parts.append(sample)
            total += count

    if total < max(1000, target_games // 2):
        print(f"ERROR: only fetched {total} games, far too few to train on", flush=True)
        raise SystemExit(2)

    sample = "\n\n".join(parts)
    if not sample.endswith("\n\n"):
        sample += "\n\n"
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(sample)
    print(f"wrote {total} games ({os.path.getsize(OUT_FILE) / 1e6:.1f} MB) to {OUT_FILE}", flush=True)
    if total < target_games:
        print(f"WARNING: only {total} games, wanted {target_games}", flush=True)


if __name__ == "__main__":
    main(
        int(sys.argv[1]) if len(sys.argv) > 1 else 25000,
        int(sys.argv[2]) if len(sys.argv) > 2 else None,
    )
