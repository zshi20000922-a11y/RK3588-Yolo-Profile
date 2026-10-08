#!/usr/bin/env python3
"""Run one RKVS camera configuration and capture board resource/result traces."""
import argparse
import json
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path


def unix_request(path, command="status"):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(2)
        client.connect(path)
        client.sendall((json.dumps({"command": command}) + "\n").encode())
        chunks = []
        while True:
            block = client.recv(65536)
            if not block:
                break
            chunks.append(block)
    return json.loads(b"".join(chunks))


def proc_sample(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().split()
    total = sum(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:]))
    status = Path(f"/proc/{pid}/status").read_text().splitlines()
    rss = next(int(x.split()[1]) for x in status if x.startswith("VmRSS:"))
    return int(fields[13]) + int(fields[14]), total, rss


def read_results(path, output, stop):
    while not stop.is_set() and not Path(path).exists():
        time.sleep(0.05)
    if stop.is_set():
        return
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client, open(output, "w") as stream:
            client.settimeout(1)
            client.connect(path)
            pending = b""
            while not stop.is_set():
                try:
                    block = client.recv(65536)
                except socket.timeout:
                    continue
                if not block:
                    break
                pending += block
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if line:
                        stream.write(line.decode(errors="replace") + "\n")
    except (OSError, TimeoutError):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--warmup", type=float, default=5)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    control = "/data/local/tmp/rkvs-cam/control.sock"
    results = "/data/local/tmp/rkvs-cam/results.sock"
    for path in (control, results):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    log = open(out / "service.log", "w")
    process = subprocess.Popen([args.service, "--config", args.config], stdout=log, stderr=subprocess.STDOUT)
    stop = threading.Event()
    reader = threading.Thread(target=read_results, args=(results, out / "results.jsonl", stop), daemon=True)
    reader.start()
    try:
        deadline = time.monotonic() + 10
        while not Path(control).exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"service exited {process.returncode}")
            time.sleep(0.1)
        time.sleep(args.warmup)
        start_status = unix_request(control)
        (out / "status_start.json").write_text(json.dumps(start_status, indent=2))
        samples = []
        started = time.monotonic()
        previous_proc, previous_total, _ = proc_sample(process.pid)
        previous_time = started
        while time.monotonic() - started < args.seconds:
            time.sleep(1)
            now = time.monotonic()
            current_proc, current_total, rss = proc_sample(process.pid)
            ncpu = os.cpu_count() or 1
            cpu = 100.0 * (current_proc - previous_proc) * ncpu / max(1, current_total - previous_total)
            load = Path("/sys/kernel/debug/rknpu/load").read_text().strip()
            cores = [int(part.split(":")[1].strip().rstrip("%")) for part in load.replace("NPU load:", "").strip(" ,").split(",")]
            temp = max(int(p.read_text()) for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"))
            freq = int(Path("/sys/class/devfreq/fdab0000.npu/cur_freq").read_text())
            samples.append({"elapsed_s": now - started, "cpu_pct": cpu, "rss_kb": rss,
                            "npu_core_pct": cores, "temp_mC": temp, "npu_freq_hz": freq})
            previous_proc, previous_total, previous_time = current_proc, current_total, now
        end_status = unix_request(control)
        (out / "status_end.json").write_text(json.dumps(end_status, indent=2))
        with open(out / "resources.jsonl", "w") as stream:
            for sample in samples:
                stream.write(json.dumps(sample) + "\n")
    finally:
        stop.set()
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        reader.join(timeout=2)
        log.close()


if __name__ == "__main__":
    main()
