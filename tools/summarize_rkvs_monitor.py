#!/usr/bin/env python3
import json
import statistics
import sys
from pathlib import Path


def q(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * p))]


def main():
    if len(sys.argv) != 2:
        print("usage: summarize_rkvs_monitor.py STATUS_JSONL", file=sys.stderr)
        return 2
    rows = []
    for line in Path(sys.argv[1]).read_text(errors="ignore").splitlines():
        try:
            outer = json.loads(line)
            status = outer.get("status", {})
            if status.get("ok"):
                rows.append((outer.get("ts", ""), status))
        except Exception:
            pass
    if not rows:
        print("No valid status rows")
        return 1

    print("# rk_vision_service Monitor Summary\n")
    print(f"- Samples: {len(rows)}")
    print(f"- Time range: {rows[0][0]} to {rows[-1][0]}")
    rss = [r[1].get("rss_kb", 0) / 1024 for r in rows]
    print(f"- RSS MB avg/p95/max: {statistics.mean(rss):.1f} / {q(rss, 0.95):.1f} / {max(rss):.1f}")
    print(f"- NPU cores: {rows[-1][1].get('npu_cores', [])}")

    source_ids = sorted({s.get("id") for _, row in rows for s in row.get("sources", [])})
    print("\n## Sources\n")
    print("| source | online% | capture fps avg | detect fps avg | latency p95 avg | errors | reconnects |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for source_id in source_ids:
        samples = [s for _, row in rows for s in row.get("sources", []) if s.get("id") == source_id]
        online = 100.0 * sum(1 for s in samples if s.get("online")) / len(samples)
        cap = [float(s.get("capture_fps", 0)) for s in samples]
        det = [float(s.get("detection_fps", 0)) for s in samples]
        lat = [float(s.get("latency_p95_ms", 0)) for s in samples]
        errors = max(int(s.get("detection_errors", 0)) for s in samples)
        reconnects = max(int(s.get("reconnects", 0)) for s in samples)
        print(f"| {source_id} | {online:.1f} | {statistics.mean(cap):.2f} | "
              f"{statistics.mean(det):.2f} | {statistics.mean(lat):.2f} | {errors} | {reconnects} |")

    print("\n## Encoders\n")
    print("| encoder | encoded max | dropped max | mpp ms avg |")
    print("|---|---:|---:|---:|")
    encoder_ids = sorted({e.get("id") for _, row in rows for e in row.get("encoders", [])})
    for encoder_id in encoder_ids:
        samples = [e for _, row in rows for e in row.get("encoders", []) if e.get("id") == encoder_id]
        print(f"| {encoder_id} | {max(int(e.get('encoded', 0)) for e in samples)} | "
              f"{max(int(e.get('dropped', 0)) for e in samples)} | "
              f"{statistics.mean(float(e.get('mpp_ms', 0)) for e in samples):.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
