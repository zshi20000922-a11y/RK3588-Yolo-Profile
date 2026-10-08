#!/usr/bin/env python3
"""Run repeatable YOLO26 document-method RK3588 benchmarks."""
import argparse, json, re, subprocess, time
from pathlib import Path

MODELS = ("yolo26n", "yolo26s")

def adb(serial, *args, check=True):
    return subprocess.run(["adb", "-s", serial, *args], check=check, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def monitor(serial, path):
    command = ("while true; do date +%s.%N; cat /sys/kernel/debug/rknpu/load 2>/dev/null; "
               "for f in /sys/class/thermal/thermal_zone*/temp /sys/class/devfreq/*npu*/cur_freq; "
               "do echo -n \"$f=\"; cat $f 2>/dev/null; done; cat /proc/loadavg; sleep 1; done")
    return subprocess.Popen(["adb", "-s", serial, "shell", command], text=True,
                            stdout=path.open("w"), stderr=subprocess.STDOUT)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--serial", required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--seconds", type=float, default=60); p.add_argument("--iterations", type=int, default=3000)
    a=p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    remote="/data/local/tmp/yolo26_document_method"
    for rnd in range(1, a.rounds+1):
        models=MODELS if rnd%2 else tuple(reversed(MODELS))
        for model in models:
            for mode in ("latency", "throughput"):
                out=a.output/model/mode/f"round_{rnd}"; out.mkdir(parents=True, exist_ok=True)
                remote_csv=f"{remote}/{model}_{mode}_r{rnd}.csv"
                cmd=(f"cd {remote} && ./rk3588_yolo_benchmark --model {model}_i8.rknn --manifest manifest.yaml "
                     f"--labels /data/local/tmp/rk_yolo_bench/labels.txt --family yolo26 "
                     f"--nv12 /data/local/tmp/rk_yolo_bench/frame.nv12 --width 1920 --height 1080 "
                     f"--mode {mode} --input-mode dma-replay --warmup 100 --iterations {a.iterations} "
                     f"--seconds {a.seconds} --output {remote_csv}")
                mon=monitor(a.serial, out/"monitor.log"); started=time.time()
                result=adb(a.serial,"shell",cmd,check=False); mon.terminate(); mon.wait(timeout=5)
                (out/"stdout.log").write_text(result.stdout)
                if result.returncode: raise RuntimeError(result.stdout)
                adb(a.serial,"pull",remote_csv,str(out/"raw_timings.csv"))
                match=re.search(r"frames=(\d+) seconds=([0-9.]+) fps=([0-9.]+).*cpu_pct=([0-9.]+)",result.stdout)
                data={"model":model,"mode":mode,"round":rnd,"started":started,
                      "contexts":1 if mode=="latency" else 3,"cores":[0] if mode=="latency" else [0,1,2]}
                if match: data.update(frames=int(match[1]),seconds=float(match[2]),fps=float(match[3]),cpu_pct=float(match[4]))
                (out/"run.json").write_text(json.dumps(data,indent=2))
                print(model,mode,rnd,result.stdout.strip(),flush=True)

if __name__ == "__main__": main()
