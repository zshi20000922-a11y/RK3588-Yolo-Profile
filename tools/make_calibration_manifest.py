#!/usr/bin/env python3
"""Create a deterministic image manifest and reject overlap with a test list."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True,
                        help="One image path per line or benchmark rows ending in image_id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    images = sorted(args.images.glob("*.jpg"), key=lambda p: p.name)
    if not images:
        raise SystemExit(f"no .jpg images found under {args.images}")
    test_ids = set()
    for line in args.test_list.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        # Benchmark rows have 4 fields; plain manifests contain only a path.
        test_ids.add(int(fields[3]) if len(fields) >= 4 else int(Path(fields[0]).stem))
    image_ids = {int(path.stem) for path in images}
    overlap = image_ids & test_ids
    if overlap:
        raise SystemExit(f"calibration/test overlap: {sorted(overlap)[:10]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{path.resolve()}\n" for path in images)
    args.output.write_text(payload)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    print(json.dumps({"images": len(images), "test_overlap": 0,
                      "manifest_sha256": digest, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
