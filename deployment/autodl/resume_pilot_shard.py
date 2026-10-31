#!/usr/bin/env python3
"""Resume retained ranges of the pinned final pilot shard; verify before use."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import os
from pathlib import Path
import time
import urllib.request

SIZE = 4167219824
PART_SIZE = SIZE // 8
SHA256 = "62aef67ee25767082503d0fade7c432c1d9425206d0824e9a812af5b9148948e"
NAME = "model-00004-of-00013.safetensors"
URL = ("https://hf-mirror.com/autotrust/JEV-27B/resolve/"
       "14b425269ed24452e0a5ad5a8f3e51d139647a88/" + NAME)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def request_range(start, end):
    request = urllib.request.Request(
        URL + f"?download=true&start={start}&end={end}&fresh={time.time_ns()}",
        headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    response = urllib.request.urlopen(request, timeout=45)
    expected = f"bytes {start}-{end}/{SIZE}"
    if response.status != 206 or response.headers.get("Content-Range") != expected:
        response.close()
        raise ValueError("Server did not honor the exact requested range")
    return response


def resume_part(index, folder, deadline):
    path = folder / f"part{index}"
    base, end = index * PART_SIZE, (index + 1) * PART_SIZE - 1
    verified_prefix = False
    for attempt in range(60):
        if time.time() >= deadline:
            raise TimeoutError("Download cutoff reached; retaining partial ranges")
        length = path.stat().st_size if path.exists() else 0
        if length > PART_SIZE:
            raise ValueError(f"Unexpected retained size: {path}")
        try:
            if length and not verified_prefix:
                count = min(length, 4096)
                with request_range(base, base + count - 1) as response:
                    remote = response.read(count + 1)
                with path.open("rb") as handle:
                    if remote != handle.read(count):
                        raise ValueError(f"Retained range has wrong origin: {path}")
                verified_prefix = True
            if length == PART_SIZE:
                print(f"PART_COMPLETE {index} {length}", flush=True)
                return path
            print(f"PART_RESUME {index} {length}/{PART_SIZE}", flush=True)
            with request_range(base + length, end) as response, path.open("ab") as handle:
                remaining = PART_SIZE - length
                while remaining:
                    if time.time() >= deadline:
                        raise TimeoutError("Download cutoff reached")
                    block = response.read(min(1024 * 1024, remaining))
                    if not block:
                        raise OSError("Incomplete range response")
                    handle.write(block)
                    remaining -= len(block)
                handle.flush()
                os.fsync(handle.fileno())
            print(f"PART_COMPLETE {index} {PART_SIZE}", flush=True)
            return path
        except ValueError:
            raise
        except Exception as error:
            print(f"PART_RETRY {index} {attempt + 1} {type(error).__name__}: {error}", flush=True)
            time.sleep(min(10, max(0, deadline - time.time())))
    raise RuntimeError(f"Retry limit for range {index}")


def main(args):
    cutoff = dt.datetime.fromisoformat(args.stop_at)
    if cutoff.tzinfo is None:
        raise ValueError("Timezone-aware cutoff required")
    target = args.model_dir / NAME
    if target.exists():
        if target.stat().st_size != SIZE or digest(target) != SHA256:
            raise ValueError("Existing target does not match; refusing to overwrite")
        print("PINNED_SHARD_VERIFIED", flush=True)
        return
    args.parts.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda i: resume_part(i, args.parts, cutoff.timestamp()), range(8)))
    staging = target.with_suffix(target.suffix + ".assembling")
    with staging.open("wb") as output:
        for path in paths:
            with path.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
        output.flush()
        os.fsync(output.fileno())
    actual = digest(staging)
    if staging.stat().st_size != SIZE or actual != SHA256:
        raise ValueError(f"Final shard verification failed: {actual}")
    os.replace(staging, target)
    print(f"PINNED_SHARD_VERIFIED {actual}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=Path, default=Path("/root/parts4"))
    parser.add_argument("--model-dir", type=Path, default=Path("/root/autodl-tmp/JEV-27B"))
    parser.add_argument("--stop-at", required=True)
    main(parser.parse_args())
