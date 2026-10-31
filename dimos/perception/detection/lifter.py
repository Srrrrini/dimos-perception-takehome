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

"""Replaceable strategies for lifting image detections into 3D."""

from abc import ABC, abstractmethod

from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D
from dimos.perception.detection.type.detection3d.imageDetections3DPC import ImageDetections3DPC
from dimos.perception.detection.type.detection3d.pointcloud_filters import PointCloudFilter


class Detection3DLifter(ABC):
    """Convert one frame of 2D detections into world-frame 3D observations."""

    @abstractmethod
    def lift(
        self,
        detections: ImageDetections2D,
        pointcloud: PointCloud2,
        camera_info: CameraInfo,
        world_to_optical_transform: Transform,
    ) -> ImageDetections3DPC: ...


class ProjectedPointCloudLifter(Detection3DLifter):
    """Project a cloud once, select points by mask or box, then apply point filters."""

    def __init__(self, filters: list[PointCloudFilter] | None = None) -> None:
        self.filters = filters

    def lift(
        self,
        detections: ImageDetections2D,
        pointcloud: PointCloud2,
        camera_info: CameraInfo,
        world_to_optical_transform: Transform,
    ) -> ImageDetections3DPC:
        return ImageDetections3DPC.from_2d(
            detections,
            pointcloud,
            camera_info,
            world_to_optical_transform,
            self.filters,
        )
