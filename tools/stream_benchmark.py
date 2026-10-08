#!/usr/bin/env python3
"""Measure client-visible stream startup, throughput and receive jitter."""
import argparse
import json
import os
import statistics
import time


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--transport", choices=("udp", "tcp"), default="udp")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{args.transport}|fflags;nobuffer|flags;low_delay|"
        "probesize;32|analyzeduration;0|max_delay;0"
    )
    import cv2

    opened_at = time.monotonic()
    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.url}")
    advertised_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_at = None
    arrivals = []
    read_ms = []
    deadline = opened_at + args.seconds + 8.0
    while time.monotonic() < deadline:
        before = time.monotonic()
        ok, _ = cap.read()
        after = time.monotonic()
        if not ok:
            if first_at is None:
                continue
            break
        if first_at is None:
            first_at = after
            deadline = after + args.seconds
        arrivals.append(after)
        read_ms.append((after - before) * 1000.0)
    cap.release()
    if first_at is None or len(arrivals) < 2:
        raise SystemExit("no decoded frames")
    intervals = [(b - a) * 1000.0 for a, b in zip(arrivals, arrivals[1:])]
    duration = arrivals[-1] - arrivals[0]
    result = {
        "url": args.url,
        "transport": args.transport,
        "stream_width": width,
        "stream_height": height,
        "advertised_fps": advertised_fps,
        "startup_to_first_frame_ms": (first_at - opened_at) * 1000.0,
        "frames": len(arrivals),
        "measurement_s": duration,
        "decoded_fps": (len(arrivals) - 1) / duration,
        "interarrival_ms": {
            "mean": statistics.fmean(intervals),
            "p50": percentile(intervals, 50),
            "p95": percentile(intervals, 95),
            "p99": percentile(intervals, 99),
            "max": max(intervals),
            "stalls_over_50ms": sum(value > 50 for value in intervals),
        },
        "read_call_ms": {
            "mean": statistics.fmean(read_ms),
            "p95": percentile(read_ms, 95),
        },
        "note": "Client receive/decode timing; not synchronized glass-to-glass latency.",
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(payload + "\n")


if __name__ == "__main__":
    main()
