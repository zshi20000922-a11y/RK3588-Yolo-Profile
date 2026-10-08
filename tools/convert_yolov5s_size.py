#!/usr/bin/env python3
"""Convert the Rockchip YOLOv5s-ReLU ONNX model at a fixed square size."""

import argparse
from rknn.api import RKNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    if args.size < 320 or args.size % 32:
        parser.error("--size must be >=320 and divisible by 32")

    rknn = RKNN(verbose=True)
    try:
        ret = rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform="rk3588",
            optimization_level=3,
        )
        if ret:
            raise RuntimeError(f"rknn.config failed: {ret}")
        ret = rknn.load_onnx(
            model=args.onnx,
            inputs=["images"],
            input_size_list=[[1, 3, args.size, args.size]],
        )
        if ret:
            raise RuntimeError(f"rknn.load_onnx failed: {ret}")
        ret = rknn.build(do_quantization=True, dataset=args.dataset)
        if ret:
            raise RuntimeError(f"rknn.build failed: {ret}")
        ret = rknn.export_rknn(args.output)
        if ret:
            raise RuntimeError(f"rknn.export_rknn failed: {ret}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
