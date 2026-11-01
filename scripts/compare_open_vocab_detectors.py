#!/usr/bin/env python3
"""Compare one detector on the fixed E12 replay-frame sample."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sqlite3
from statistics import mean, median
from time import perf_counter
from typing import Protocol, cast

from dimos.memory.codecs.jpeg import JpegCodec
from dimos.msgs.sensor_msgs.Image import Image
from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D

FRAME_IDS = (50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800)
MODEL_NAMES = {
    "coco-yolo": "yolo11n-seg.pt",
    "yoloe": "yoloe-11s-seg-pf.pt",
    "grounding-dino": "IDEA-Research/grounding-dino-tiny",
}


class Predictor(Protocol):
    def __call__(self, image: Image) -> ImageDetections2D: ...


def detector_predictor(name: str, device: str | None) -> Predictor:
    if name == "coco-yolo":
        from dimos.perception.detection.detectors.yolo import Yolo11Seg2DDetector

        return Yolo11Seg2DDetector(device=device).process_image
    if name == "yoloe":
        from dimos.perception.detection.detectors.yoloe import Yoloe2DDetector

        return Yoloe2DDetector(device=device).predict_image

    from dimos.perception.detection.detectors.grounding_dino import GroundingDino2DDetector

    return GroundingDino2DDetector().predict_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector", choices=("coco-yolo", "yoloe", "grounding-dino"), required=True
    )
    parser.add_argument("--database", type=Path, default=Path("data/go2_short.db"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    if args.detector == "grounding-dino" and args.device is not None:
        parser.error("--device is not yet configurable for grounding-dino")
    predict = detector_predictor(args.detector, args.device)
    counts: Counter[str] = Counter()
    latencies_ms: list[float] = []

    with sqlite3.connect(args.database) as database:
        for frame_id in FRAME_IDS:
            row = database.execute(
                "SELECT data FROM color_image_blob WHERE id = ?", (frame_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError(f"missing color_image_blob row {frame_id}")
            image = JpegCodec().decode(cast("bytes", row[0]))
            started = perf_counter()
            detections = predict(image)
            latencies_ms.append((perf_counter() - started) * 1_000.0)
            counts.update(detection.name for detection in detections)

    warm_latencies = latencies_ms[1:]
    p95_index = max(0, int(0.95 * len(warm_latencies) + 0.999999) - 1)
    sorted_warm = sorted(warm_latencies)

    print(
        json.dumps(
            {
                "detector": args.detector,
                "model": MODEL_NAMES[args.detector],
                "database_sha256": hashlib.sha256(args.database.read_bytes()).hexdigest(),
                "device": args.device or "auto",
                "frame_ids": FRAME_IDS,
                "frames": len(FRAME_IDS),
                "labels": dict(sorted(counts.items())),
                "latency_ms_cold": round(latencies_ms[0], 2),
                "latency_ms_mean_warm": round(mean(warm_latencies), 2),
                "latency_ms_median_warm": round(median(warm_latencies), 2),
                "latency_ms_p95_warm": round(sorted_warm[p95_index], 2),
                "throughput_hz_warm": round(1_000.0 / mean(warm_latencies), 2),
                "platform": platform.platform(),
                "total": sum(counts.values()),
                "versions": {
                    "torch": version("torch"),
                    "ultralytics": version("ultralytics"),
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
