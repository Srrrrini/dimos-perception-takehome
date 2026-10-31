# Copyright 2025-2026 Dimensional Inc.
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


from collections.abc import Callable
import math
from threading import RLock
import time

from reactivex import operators as ops
from reactivex.observable import Observable

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.vision_msgs.Detection2DArray import Detection2DArray
from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray
from dimos.perception.detection.lifter import Detection3DLifter, ProjectedPointCloudLifter
from dimos.perception.detection.module2D import Config as Detection2DConfig, Detection2DModule
from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D
from dimos.perception.detection.type.detection3d.imageDetections3DPC import ImageDetections3DPC
from dimos.types.timestamped import align_timestamped
from dimos.utils.logging_config import setup_logger
from dimos.utils.reactive import backpressure

logger = setup_logger()


class Detection3DConfig(Detection2DConfig):
    """Configuration with a replaceable camera-LiDAR lifting strategy."""

    lifter: Callable[[], Detection3DLifter] = ProjectedPointCloudLifter


class Detection3DModule(Detection2DModule):
    config: Detection3DConfig

    color_image: In[Image]
    pointcloud: In[PointCloud2]

    detections: Out[Detection2DArray]
    detections_3d: Out[Detection3DArray]
    # just for visualization,
    # emits latest pointclouds of detected objects in a frame
    detected_pointcloud_0: Out[PointCloud2]
    detected_pointcloud_1: Out[PointCloud2]
    detected_pointcloud_2: Out[PointCloud2]

    # just for visualization, emits latest top 3 detections in a frame
    detected_image_0: Out[Image]
    detected_image_1: Out[Image]
    detected_image_2: Out[Image]

    detection_3d_stream: Observable[ImageDetections3DPC] | None = None

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.lifter = self.config.lifter()
        self._metrics_lock = RLock()
        self._metrics_started_at = time.perf_counter()
        self._input_frames = 0
        self._first_input_ts: float | None = None
        self._last_input_ts: float | None = None
        self._inference_calls = 0
        self._detections_2d = 0
        self._inference_ms: list[float] = []
        self._fusion_calls = 0
        self._fusion_detections_2d = 0
        self._fusion_detections_3d = 0
        self._missing_transforms = 0
        self._alignment_ms: list[float] = []
        self._fusion_ms: list[float] = []

    def _observe_input(self, image: Image) -> None:
        timestamp = float(image.ts)
        with self._metrics_lock:
            self._input_frames += 1
            if self._first_input_ts is None:
                self._first_input_ts = timestamp
            self._last_input_ts = timestamp

    def process_image_frame(self, image: Image) -> ImageDetections2D:
        """Run the configured detector and record its effective replay cost."""
        started_at = time.perf_counter()
        detections = super().process_image_frame(image)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        with self._metrics_lock:
            self._inference_calls += 1
            self._detections_2d += len(detections)
            self._inference_ms.append(elapsed_ms)
        return detections

    def process_frame(
        self,
        detections: ImageDetections2D,
        pointcloud: PointCloud2,
        transform: Transform | None,
    ) -> ImageDetections3DPC:
        started_at = time.perf_counter()
        if not transform:
            with self._metrics_lock:
                self._fusion_calls += 1
                self._fusion_detections_2d += len(detections)
                self._missing_transforms += 1
                self._fusion_ms.append((time.perf_counter() - started_at) * 1000.0)
            return ImageDetections3DPC(detections.image, [])

        result = self.lifter.lift(detections, pointcloud, self.config.camera_info, transform)
        alignment_ms = abs(float(detections.image.ts) - float(pointcloud.ts)) * 1000.0
        with self._metrics_lock:
            self._fusion_calls += 1
            self._fusion_detections_2d += len(detections)
            self._fusion_detections_3d += len(result)
            self._alignment_ms.append(alignment_ms)
            self._fusion_ms.append((time.perf_counter() - started_at) * 1000.0)
        return result

    @staticmethod
    def _distribution(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        ordered = sorted(values)

        def percentile(fraction: float) -> float:
            index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
            return ordered[index]

        return {
            "mean": sum(ordered) / len(ordered),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "max": ordered[-1],
        }

    @rpc
    def get_perception_metrics(self) -> dict[str, object]:
        """Return detector and 2D-to-3D replay throughput metrics."""
        with self._metrics_lock:
            elapsed_s = max(time.perf_counter() - self._metrics_started_at, 1e-9)
            sensor_span_s = (
                self._last_input_ts - self._first_input_ts
                if self._first_input_ts is not None and self._last_input_ts is not None
                else 0.0
            )
            return {
                "input_frames": self._input_frames,
                "sensor_span_s": sensor_span_s,
                "source_fps": self._input_frames / sensor_span_s if sensor_span_s > 0 else 0.0,
                "inference_calls": self._inference_calls,
                "effective_inference_hz": self._inference_calls / elapsed_s,
                "detections_2d": self._detections_2d,
                "inference_ms": self._distribution(self._inference_ms),
                "fusion_calls": self._fusion_calls,
                "fusion_detections_2d": self._fusion_detections_2d,
                "fusion_detections_3d": self._fusion_detections_3d,
                "fusion_survival_ratio": (
                    self._fusion_detections_3d / self._fusion_detections_2d
                    if self._fusion_detections_2d
                    else 0.0
                ),
                "missing_transforms": self._missing_transforms,
                "alignment_ms": self._distribution(self._alignment_ms),
                "fusion_ms": self._distribution(self._fusion_ms),
                "lifter": type(self.lifter).__name__,
            }

    def pixel_to_3d(
        self,
        pixel: tuple[int, int],
        assumed_depth: float = 1.0,
    ) -> Vector3:
        """Unproject 2D pixel coordinates to 3D position in camera optical frame.

        Args:
            camera_info: Camera calibration information
            assumed_depth: Assumed depth in meters (default 1.0m from camera)

        Returns:
            Vector3 position in camera optical frame coordinates
        """
        # Extract camera intrinsics
        fx, fy = self.config.camera_info.K[0], self.config.camera_info.K[4]
        cx, cy = self.config.camera_info.K[2], self.config.camera_info.K[5]

        # Unproject pixel to normalized camera coordinates
        x_norm = (pixel[0] - cx) / fx
        y_norm = (pixel[1] - cy) / fy

        # Create 3D point at assumed depth in camera optical frame
        # Camera optical frame: X right, Y down, Z forward
        return Vector3(x_norm * assumed_depth, y_norm * assumed_depth, assumed_depth)

    @skill
    def ask_vlm(self, question: str) -> str:
        """asks a visual model about the view of the robot, for example
        is the bannana in the trunk?
        """
        from dimos.models.vl.qwen import QwenVlModel

        model = QwenVlModel()
        image = self.color_image.get_next()
        return model.query(image, question)

    # @skill
    @rpc
    def nav_vlm(self, question: str) -> str:
        """
        query visual model about the view in front of the camera
        you can ask to mark objects like:

        "red cup on the table left of the pencil"
        "laptop on the desk"
        "a person wearing a red shirt"
        """
        from dimos.models.vl.qwen import QwenVlModel

        model = QwenVlModel()
        image = self.color_image.get_next()
        result = model.query_detections(image, question)

        print("VLM result:", result, "for", image, "and question", question)

        if isinstance(result, str) or not result or not len(result):
            return None  # type: ignore[return-value]

        detections: ImageDetections2D = result

        print(detections)
        if not len(detections):
            print("No 2d detections")
            return None  # type: ignore[return-value]

        pc = self.pointcloud.get_next()
        transform = self.tfbuffer.get("camera_optical", pc.frame_id, detections.image.ts, 5.0)

        detections3d = self.process_frame(detections, pc, transform)

        if len(detections3d):
            return detections3d[0].pose  # type: ignore[no-any-return]
        print("No 3d detections, projecting 2d")

        center = detections[0].get_bbox_center()
        return PoseStamped(
            ts=detections.image.ts,
            frame_id="world",
            position=self.pixel_to_3d(center, assumed_depth=1.5),
            orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        )

    @rpc
    def start(self) -> None:
        super().start()
        self.color_image.subscribe(self._observe_input)

        def detection2d_to_3d(args):  # type: ignore[no-untyped-def]
            detections, pc = args
            transform = self.tfbuffer.get("camera_optical", pc.frame_id, detections.image.ts, 5.0)
            return self.process_frame(detections, pc, transform)

        self.detection_stream_3d = align_timestamped(  # type: ignore[type-var]
            backpressure(self.detection_stream_2d()),
            self.pointcloud.observable(),
            match_tolerance=0.25,
            buffer_size=20.0,
        ).pipe(ops.map(detection2d_to_3d))

        self.detection_stream_3d.subscribe(self._publish_detections)

    @rpc
    def stop(self) -> None:
        logger.info("Perception replay metrics", metrics=self.get_perception_metrics())
        super().stop()

    def _publish_detections(self, detections: ImageDetections3DPC) -> None:
        if not detections:
            return

        self.detections_3d.publish(detections.to_ros_detection3d_array())
        for index, detection in enumerate(detections[:3]):
            pointcloud_topic = getattr(self, "detected_pointcloud_" + str(index))
            pointcloud_topic.publish(detection.pointcloud)
