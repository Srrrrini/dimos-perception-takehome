#!/usr/bin/env python3
"""Compare one detector on the fixed E12 replay-frame sample."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
from typing import Protocol, cast

from dimos.memory.codecs.jpeg import JpegCodec
from dimos.msgs.sensor_msgs.Image import Image
from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D

FRAME_IDS = (50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800)


class Predictor(Protocol):
    def __call__(self, image: Image) -> ImageDetections2D: ...


def detector_predictor(name: str) -> Predictor:
    if name == "coco-yolo":
        from dimos.perception.detection.detectors.yolo import Yolo11Seg2DDetector

        return Yolo11Seg2DDetector().process_image
    if name == "yoloe":
        from dimos.perception.detection.detectors.yoloe import Yoloe2DDetector

        return Yoloe2DDetector().predict_image

    from dimos.perception.detection.detectors.grounding_dino import GroundingDino2DDetector

    return GroundingDino2DDetector().predict_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=("coco-yolo", "yoloe", "grounding-dino"), required=True)
    parser.add_argument("--database", type=Path, default=Path("data/go2_short.db"))
    args = parser.parse_args()
    predict = detector_predictor(args.detector)
    counts: Counter[str] = Counter()

    with sqlite3.connect(args.database) as database:
        for frame_id in FRAME_IDS:
            row = database.execute(
                "SELECT data FROM color_image_blob WHERE id = ?", (frame_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError(f"missing color_image_blob row {frame_id}")
            image = JpegCodec().decode(cast("bytes", row[0]))
            counts.update(detection.name for detection in predict(image))

    print(
        json.dumps(
            {
                "detector": args.detector,
                "frame_ids": FRAME_IDS,
                "frames": len(FRAME_IDS),
                "labels": dict(sorted(counts.items())),
                "total": sum(counts.values()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
