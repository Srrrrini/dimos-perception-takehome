# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from dimos.perception.detection.detectors.yoloe import Yoloe2DDetector


def test_yoloe_falls_back_to_first_run_download_when_lfs_is_unavailable() -> None:
    yoloe = MagicMock()
    fake_ultralytics = SimpleNamespace(YOLOE=yoloe)

    with (
        patch.dict(sys.modules, {"ultralytics": fake_ultralytics}),
        patch(
            "dimos.perception.detection.detectors.yoloe.get_data",
            side_effect=FileNotFoundError("no LFS archive"),
        ),
    ):
        Yoloe2DDetector(device="cpu")

    yoloe.assert_called_once_with("yoloe-11s-seg-pf.pt")
