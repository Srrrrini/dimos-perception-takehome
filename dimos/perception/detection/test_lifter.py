# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, patch

import numpy as np

from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.perception.detection.lifter import (
    Detection3DLifter,
    HybridMaskBoxLifter,
    ProjectedPointCloudLifter,
)
from dimos.perception.detection.module2D import Detection2DModule
from dimos.perception.detection.module3D import Detection3DModule
from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D
from dimos.perception.detection.type.detection2d.seg import Detection2DSeg
from dimos.perception.detection.type.detection3d.imageDetections3DPC import ImageDetections3DPC
from dimos.perception.detection.type.detection3d.pointcloud import Detection3DPC
from dimos.robot.unitree.go2.connection import GO2Connection


def test_projected_lifter_delegates_to_batch_projection() -> None:
    lifter = ProjectedPointCloudLifter()
    detections = MagicMock(spec=ImageDetections2D)
    cloud = MagicMock(spec=PointCloud2)
    transform = MagicMock(spec=Transform)
    expected = MagicMock(spec=ImageDetections3DPC)

    with patch.object(ImageDetections3DPC, "from_2d", return_value=expected) as from_2d:
        actual = lifter.lift(detections, cloud, GO2Connection.camera_info_static, transform)

    assert actual is expected
    from_2d.assert_called_once_with(
        detections,
        cloud,
        GO2Connection.camera_info_static,
        transform,
        None,
    )


def test_detection_module_accepts_injected_lifter() -> None:
    lifter = MagicMock(spec=Detection3DLifter)
    module = Detection3DModule(
        camera_info=GO2Connection.camera_info_static,
        detector=MagicMock,
        lifter=lambda: lifter,
    )

    assert module.lifter is lifter
    assert module.get_perception_metrics()["lifter"] == type(lifter).__name__

    module._close_module()


def test_hybrid_lifter_recovers_sparse_mask_with_box() -> None:
    image = MagicMock()
    detection = MagicMock(spec=Detection2DSeg)
    detection.mask = np.ones((8, 8), dtype=np.uint8)
    detection.bbox = (1.0, 1.0, 6.0, 6.0)
    detection.track_id = 3
    detection.class_id = 1
    detection.confidence = 0.8
    detection.name = "cup"
    detection.ts = 1.0
    detection.image = image
    detections = ImageDetections2D(image=image, detections=[detection])
    cloud = MagicMock(spec=PointCloud2)
    cloud.frame_id = "world"
    cloud.ts = 1.0
    transform = MagicMock(spec=Transform)
    fallback = MagicMock(spec=Detection3DPC)
    fallback.ts = 1.0
    lifter = HybridMaskBoxLifter(min_mask_projected_points=20)

    with (
        patch.object(
            Detection3DPC,
            "project_cloud",
            return_value=(np.zeros((1, 3)), np.zeros((1, 2))),
        ),
        patch.object(lifter, "_count_mask_points", return_value=1),
        patch.object(Detection3DPC, "from_projection", return_value=fallback) as project,
    ):
        result = lifter.lift(detections, cloud, GO2Connection.camera_info_static, transform)

    assert result.detections == [fallback]
    assert not hasattr(project.call_args.args[0], "mask")
    assert lifter.get_metrics()["fallback_recovered"] == 1


def test_detector_exception_is_isolated_to_one_frame() -> None:
    image = MagicMock()
    image.ts = 1.0
    module = Detection3DModule(
        camera_info=GO2Connection.camera_info_static,
        detector=MagicMock,
    )

    with patch.object(Detection2DModule, "process_image_frame", side_effect=RuntimeError("bad")):
        result = module.process_image_frame(image)

    assert len(result) == 0
    assert module.get_perception_metrics()["inference_failures"] == 1
    module._close_module()


def test_lifter_exception_is_isolated_to_one_frame() -> None:
    image = MagicMock()
    image.ts = 1.0
    detections = ImageDetections2D(image=image, detections=[])
    cloud = MagicMock(spec=PointCloud2)
    cloud.ts = 1.0
    transform = MagicMock(spec=Transform)
    lifter = MagicMock(spec=Detection3DLifter)
    lifter.lift.side_effect = RuntimeError("bad cloud")
    module = Detection3DModule(
        camera_info=GO2Connection.camera_info_static,
        detector=MagicMock,
        lifter=lambda: lifter,
    )

    result = module.process_frame(detections, cloud, transform)

    assert len(result) == 0
    assert module.get_perception_metrics()["fusion_failures"] == 1
    module._close_module()
