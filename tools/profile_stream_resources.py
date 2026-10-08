#!/usr/bin/env python3
"""Sample detector, encoder and RTSP server load on an RK3588 board."""
import argparse
import csv
import os
import time
from pathlib import Path


GROUPS = {
    "detector": b"rknn_dual_camera_detector",
    "encoder": b"gst-launch-1.0",
    "rtsp": b"rtsp_server_ctypes.py",
}


def processes():
    found = {name: [] for name in GROUPS}
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            cmdline = (item / "cmdline").read_bytes().replace(b"\0", b" ")
            for name, needle in GROUPS.items():
                if needle in cmdline and b"profile_stream_resources.py" not in cmdline:
                    found[name].append(int(item.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return found


def process_values(pids):
    ticks = rss = 0
    for pid in pids:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().split()
            ticks += int(fields[13]) + int(fields[14])
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss += int(line.split()[1])
                    break
        except (FileNotFoundError, ProcessLookupError):
            pass
    return ticks, rss


def total_ticks():
    return sum(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:]))


def net_bytes():
    values = {}
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        name, data = line.split(":", 1)
        fields = data.split()
        values[name.strip()] = (int(fields[0]), int(fields[8]))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cpu_count = os.cpu_count() or 8
    groups = processes()
    previous = {name: process_values(pids)[0] for name, pids in groups.items()}
    previous_total = total_ticks()
    previous_net = net_bytes()
    with open(args.output, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample", "detector_cpu_pct", "detector_rss_kb",
                         "encoder_cpu_pct", "encoder_rss_kb", "rtsp_cpu_pct",
                         "rtsp_rss_kb", "network_rx_mbps", "network_tx_mbps"))
        for sample in range(args.seconds):
            time.sleep(1)
            current_total = total_ticks()
            denominator = max(1, current_total - previous_total)
            row = [sample]
            for name in ("detector", "encoder", "rtsp"):
                ticks, rss = process_values(groups[name])
                cpu = (ticks - previous[name]) * 100.0 * cpu_count / denominator
                row.extend((round(cpu, 2), rss))
                previous[name] = ticks
            current_net = net_bytes()
            rx = tx = 0
            for interface, values in current_net.items():
                if interface == "lo" or interface not in previous_net:
                    continue
                rx += max(0, values[0] - previous_net[interface][0])
                tx += max(0, values[1] - previous_net[interface][1])
            row.extend((round(rx * 8 / 1_000_000, 3),
                        round(tx * 8 / 1_000_000, 3)))
            writer.writerow(row)
            stream.flush()
            previous_total = current_total
            previous_net = current_net


if __name__ == "__main__":
    main()
