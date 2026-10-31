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
from threading import RLock

import numpy as np

from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.perception.detection.type.detection2d.bbox import Detection2DBBox
from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D
from dimos.perception.detection.type.detection3d.imageDetections3DPC import ImageDetections3DPC
from dimos.perception.detection.type.detection3d.pointcloud import Detection3DPC, lattice_quantum
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

    def get_metrics(self) -> dict[str, object]:
        """Return strategy-specific diagnostics for experiment logging."""
        return {}


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


class HybridMaskBoxLifter(Detection3DLifter):
    """Prefer precise masks, falling back to boxes when projected support is sparse."""

    def __init__(
        self,
        filters: list[PointCloudFilter] | None = None,
        min_mask_projected_points: int = 20,
    ) -> None:
        if min_mask_projected_points <= 0:
            raise ValueError("min_mask_projected_points must be positive")
        self.filters = filters
        self.min_mask_projected_points = min_mask_projected_points
        self._lock = RLock()
        self._mask_attempts = 0
        self._mask_retained = 0
        self._fallback_attempts = 0
        self._fallback_recovered = 0
        self._mask_projected_points: list[int] = []

    def lift(
        self,
        detections: ImageDetections2D,
        pointcloud: PointCloud2,
        camera_info: CameraInfo,
        world_to_optical_transform: Transform,
    ) -> ImageDetections3DPC:
        world_points, points_2d = Detection3DPC.project_cloud(
            pointcloud, camera_info, world_to_optical_transform
        )
        quantum = lattice_quantum(world_points)
        splat_m = quantum / 2 if quantum is not None else None
        lifted: list[Detection3DPC] = []

        for detection in detections:
            mask = getattr(detection, "mask", None)
            if mask is None:
                result = self._from_projection(
                    detection,
                    world_points,
                    points_2d,
                    camera_info,
                    world_to_optical_transform,
                    pointcloud,
                    splat_m,
                )
                if result is not None:
                    lifted.append(result)
                continue

            projected_points = self._count_mask_points(
                mask,
                world_points,
                points_2d,
                camera_info,
                world_to_optical_transform,
                splat_m,
            )
            with self._lock:
                self._mask_attempts += 1
                self._mask_projected_points.append(projected_points)

            if projected_points >= self.min_mask_projected_points:
                result = self._from_projection(
                    detection,
                    world_points,
                    points_2d,
                    camera_info,
                    world_to_optical_transform,
                    pointcloud,
                    splat_m,
                )
                if result is not None:
                    with self._lock:
                        self._mask_retained += 1
                    lifted.append(result)
                    continue

            with self._lock:
                self._fallback_attempts += 1
            bbox_detection = Detection2DBBox(
                bbox=detection.bbox,
                track_id=detection.track_id,
                class_id=detection.class_id,
                confidence=detection.confidence,
                name=detection.name,
                ts=detection.ts,
                image=detection.image,
            )
            fallback = self._from_projection(
                bbox_detection,
                world_points,
                points_2d,
                camera_info,
                world_to_optical_transform,
                pointcloud,
                splat_m,
            )
            if fallback is not None:
                with self._lock:
                    self._fallback_recovered += 1
                lifted.append(fallback)

        return ImageDetections3DPC(image=detections.image, detections=lifted)

    def _from_projection(
        self,
        detection: Detection2DBBox,
        world_points: np.ndarray,
        points_2d: np.ndarray,
        camera_info: CameraInfo,
        transform: Transform,
        pointcloud: PointCloud2,
        splat_m: float | None,
    ) -> Detection3DPC | None:
        return Detection3DPC.from_projection(
            detection,
            world_points,
            points_2d,
            camera_info,
            transform,
            pointcloud.frame_id,
            pointcloud.ts,
            self.filters,
            splat_m=splat_m,
        )

    @staticmethod
    def _count_mask_points(
        mask: np.ndarray,
        world_points: np.ndarray,
        points_2d: np.ndarray,
        camera_info: CameraInfo,
        transform: Transform,
        splat_m: float | None,
    ) -> int:
        if len(world_points) == 0:
            return 0
        height, width = mask.shape[:2]

        def hit(cols_f: np.ndarray, rows_f: np.ndarray) -> np.ndarray:
            cols = np.clip(cols_f.astype(int), 0, width - 1)
            rows = np.clip(rows_f.astype(int), 0, height - 1)
            return np.asarray(mask[rows, cols] > 0, dtype=bool)

        selected = hit(points_2d[:, 0], points_2d[:, 1])
        if splat_m is not None:
            camera = transform.inverse().translation.to_numpy()
            ranges = np.linalg.norm(world_points - camera, axis=1)
            radius = splat_m * camera_info.K[0] / np.maximum(ranges, 1e-6)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                selected |= hit(points_2d[:, 0] + dx * radius, points_2d[:, 1] + dy * radius)
        return int(selected.sum())

    def get_metrics(self) -> dict[str, object]:
        with self._lock:
            attempts = self._mask_attempts
            retained = self._mask_retained
            fallbacks = self._fallback_attempts
            recovered = self._fallback_recovered
            counts = list(self._mask_projected_points)
        sparse = sum(count < self.min_mask_projected_points for count in counts)
        return {
            "min_mask_projected_points": self.min_mask_projected_points,
            "mask_attempts": attempts,
            "mask_retained": retained,
            "sparse_masks": sparse,
            "sparse_mask_rate": sparse / attempts if attempts else 0.0,
            "fallback_attempts": fallbacks,
            "fallback_recovered": recovered,
            "fallback_recovery_rate": recovered / fallbacks if fallbacks else 0.0,
            "mask_projected_points_mean": sum(counts) / len(counts) if counts else 0.0,
        }
