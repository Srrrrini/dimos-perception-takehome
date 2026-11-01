# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from unittest.mock import MagicMock

from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.perception.detection.module2D import Detection2DModule


def test_module_stop_releases_detector() -> None:
    detector = MagicMock()
    module = Detection2DModule(
        camera_info=MagicMock(spec=CameraInfo),
        detector=lambda: detector,
    )

    module.stop()

    detector.stop.assert_called_once_with()
