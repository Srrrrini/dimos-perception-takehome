# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from unittest.mock import MagicMock, patch

import numpy as np

from dimos.perception.detection.detectors.grounding_dino import (
    GroundingDino2DDetector,
    GroundingDinoSam2DDetector,
)
from dimos.perception.detection.type.detection2d.bbox import Detection2DBBox
from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D
from dimos.perception.detection.type.detection2d.seg import Detection2DSeg


def test_grounding_dino_adapter_forwards_fixed_queries_and_threshold() -> None:
    image = MagicMock()
    expected = MagicMock()
    model = MagicMock()
    model.query_detections.return_value = expected

    with patch(
        "dimos.perception.detection.detectors.grounding_dino.GroundingDinoDetector",
        return_value=model,
    ):
        detector = GroundingDino2DDetector(queries=("shelf", "carton"), threshold=0.25)

    assert detector.process_image(image) is expected
    model.query_detections.assert_called_once_with(image, ["shelf", "carton"], 0.25)

    detector.stop()
    model.stop.assert_called_once_with()


def test_grounding_dino_sam_preserves_labels_and_adds_masks() -> None:
    image = MagicMock()
    image.ts = 1.0
    image.height = 8
    image.width = 10
    image.to_opencv.return_value = np.zeros((8, 10, 3), dtype=np.uint8)
    box = Detection2DBBox((1.0, 2.0, 7.0, 6.0), -1, 3, 0.8, "carton", 1.0, image)
    masks = MagicMock()
    masks.data = [MagicMock()]
    masks.data[0].detach.return_value.cpu.return_value.numpy.return_value = np.ones(
        (8, 10), dtype=np.float32
    )
    sam = MagicMock()
    sam.predict.return_value = [MagicMock(masks=masks)]
    detector = GroundingDinoSam2DDetector(device="cpu")
    detector.__dict__["_sam"] = sam

    with patch.object(
        GroundingDino2DDetector,
        "predict_image",
        return_value=ImageDetections2D(image=image, detections=[box]),
    ):
        result = detector.predict_image(image)

    assert len(result) == 1
    assert isinstance(result[0], Detection2DSeg)
    assert result[0].name == "carton"
    assert np.all(result[0].mask == 255)
    sam.predict.assert_called_once_with(
        source=image.to_opencv.return_value,
        bboxes=[box.bbox],
        device="cpu",
        verbose=False,
    )
