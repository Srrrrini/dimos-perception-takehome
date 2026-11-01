# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from unittest.mock import MagicMock, patch

from dimos.perception.detection.detectors.grounding_dino import GroundingDino2DDetector


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
