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

"""DimOS module that retains object detections in a world-frame map."""

from __future__ import annotations

from collections.abc import Callable
import math
from threading import RLock
import time

from dimos_lcm.vision_msgs import BoundingBox3D, ObjectHypothesis, ObjectHypothesisWithPose
import numpy as np
from reactivex.disposable import Disposable
from scipy.spatial.transform import Rotation

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseArray import PoseArray
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.std_msgs.Header import Header
from dimos.msgs.vision_msgs.Detection3D import Detection3D
from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray
from dimos.perception.detection.appearance import decode_detection_id
from dimos.perception.semantic_map.backend import SemanticMapBackend
from dimos.perception.semantic_map.tracker import (
    ObjectObservation,
    SemanticMapTracker,
    SemanticObject,
)
from dimos.types.timestamped import to_timestamp
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class SemanticObjectMapConfig(ModuleConfig):
    """Configuration for conservative world-space object association."""

    association_distance_m: float = 0.75
    min_observations: int = 2
    backend: Callable[[float], SemanticMapBackend] = SemanticMapTracker


class SemanticObjectMap(Module):
    """Build and query a persistent 3D semantic object map."""

    config: SemanticObjectMapConfig

    detections_3d: In[Detection3DArray]
    pgo_keyframes: In[PoseArray]
    semantic_map: Out[Detection3DArray]

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.config.min_observations <= 0:
            raise ValueError("min_observations must be positive")
        self._tracker = self.config.backend(self.config.association_distance_m)
        self._lock = RLock()
        self._metrics_started_at = time.perf_counter()
        self._frames_3d = 0
        self._observations_in = 0
        self._matched_observations = 0
        self._new_tracks = 0
        self._association_ms: list[float] = []
        self._keyframe_poses: tuple[tuple[float, ...], ...] = ()

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.detections_3d.subscribe(self._on_detections)))
        self.register_disposable(Disposable(self.pgo_keyframes.subscribe(self._on_keyframes)))

    def _on_detections(self, message: Detection3DArray) -> None:
        started_at = time.perf_counter()
        with self._lock:
            keyframe_poses = self._keyframe_poses
        observations = [
            observation
            for detection in message.detections[: message.detections_length]
            if (
                observation := self._to_observation(detection, message.ts, keyframe_poses)
            )
            is not None
        ]
        with self._lock:
            self._frames_3d += 1
            self._observations_in += len(observations)
        if not observations:
            return
        with self._lock:
            objects_before = len(self._tracker.objects)
            self._tracker.update(observations)
            created = len(self._tracker.objects) - objects_before
            self._new_tracks += created
            self._matched_observations += len(observations) - created
            self._association_ms.append((time.perf_counter() - started_at) * 1000.0)
            mapped = self._tracker.confirmed(self.config.min_observations)
        self.semantic_map.publish(self._to_message(mapped, message.ts, message.frame_id))

    def _on_keyframes(self, message: PoseArray) -> None:
        poses = tuple(
            (
                float(pose.x),
                float(pose.y),
                float(pose.z),
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            )
            for pose in message.poses
        )
        with self._lock:
            self._keyframe_poses = poses
            self._tracker.apply_keyframe_poses(poses)
            mapped = self._tracker.confirmed(self.config.min_observations)
        if mapped:
            self.semantic_map.publish(self._to_message(mapped, message.ts, message.frame_id))

    @skill
    def where_is_object(self, name: str) -> str:
        """Return remembered world coordinates for an object class.

        Args:
            name: Object label to find, such as chair, person, or potted plant.
        """
        with self._lock:
            matches = self._tracker.find(name, self.config.min_observations)
        if not matches:
            return f"No confirmed object matching '{name}' is in the semantic map."
        descriptions = [
            f"{obj.label} {obj.object_id[:8]} at "
            f"({obj.position[0]:.2f}, {obj.position[1]:.2f}, {obj.position[2]:.2f}) m "
            f"in world; observations={obj.observation_count}, confidence={obj.confidence:.2f}, "
            f"position_stddev={obj.position_stddev_m:.2f} m"
            f", state={obj.state.value}, moves={obj.move_count}"
            for obj in matches
        ]
        return "\n".join(descriptions)

    @rpc
    def get_semantic_map(self) -> Detection3DArray:
        """Return the current confirmed map for programmatic consumers."""
        with self._lock:
            objects = self._tracker.confirmed(self.config.min_observations)
        timestamp = max((obj.last_seen for obj in objects), default=0.0)
        return self._to_message(objects, timestamp, "world")

    @rpc
    def get_metrics(self) -> dict[str, object]:
        """Return replay-friendly map persistence and consistency metrics."""
        with self._lock:
            all_objects = list(self._tracker.objects)
            confirmed = self._tracker.confirmed(self.config.min_observations)
            frames_3d = self._frames_3d
            observations_in = self._observations_in
            matched_observations = self._matched_observations
            new_tracks = self._new_tracks
            association_ms = list(self._association_ms)
        observations = sum(obj.observation_count for obj in confirmed)
        deviations = [obj.position_stddev_m for obj in confirmed]
        lifetimes = [obj.last_seen - obj.first_seen for obj in confirmed]
        return {
            "candidate_objects": len(all_objects),
            "confirmed_objects": len(confirmed),
            "confirmation_ratio": len(confirmed) / len(all_objects) if all_objects else 0.0,
            "frames_3d": frames_3d,
            "observations_in": observations_in,
            "matched_observations": matched_observations,
            "new_tracks": new_tracks,
            "association_rate": matched_observations / observations_in if observations_in else 0.0,
            "new_track_rate": new_tracks / observations_in if observations_in else 0.0,
            "confirmed_observations": observations,
            "mean_observations_per_object": observations / len(confirmed) if confirmed else 0.0,
            "mean_position_stddev_m": sum(deviations) / len(deviations) if deviations else 0.0,
            "max_position_stddev_m": max(deviations, default=0.0),
            "max_observed_track_duration_s": max(lifetimes, default=0.0),
            "association_ms_mean": (
                sum(association_ms) / len(association_ms) if association_ms else 0.0
            ),
            "association_ms_p95": self._percentile(association_ms, 0.95),
            "same_label_pairs_within_0_5m": self._near_pairs(confirmed, 0.5),
            "same_label_pairs_within_0_75m": self._near_pairs(confirmed, 0.75),
            "same_label_pairs_within_1_0m": self._near_pairs(confirmed, 1.0),
            "backend": type(self._tracker).__name__,
            "active_objects": sum(obj.state.value == "active" for obj in all_objects),
            "dormant_objects": sum(obj.state.value == "dormant" for obj in all_objects),
            "moved_objects": sum(obj.state.value == "moved" for obj in all_objects),
            "move_events": sum(obj.move_count for obj in all_objects),
            "graph_corrections": self._tracker.graph_corrections,
            "graph_merges": self._tracker.graph_merges,
            "max_graph_correction_m": self._tracker.max_graph_correction_m,
        }

    @rpc
    def stop(self) -> None:
        logger.info("Semantic map replay metrics", metrics=self.get_metrics())
        super().stop()

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]

    @staticmethod
    def _near_pairs(objects: list[SemanticObject], threshold_m: float) -> int:
        """Count same-label nearby pairs as a duplicate diagnostic, not ground truth."""
        return sum(
            1
            for index, left in enumerate(objects)
            for right in objects[index + 1 :]
            if left.label.casefold() == right.label.casefold()
            and math.dist(left.position, right.position) <= threshold_m
        )

    @staticmethod
    def _to_observation(
        detection: Detection3D,
        fallback_ts: float,
        keyframe_poses: tuple[tuple[float, ...], ...] = (),
    ) -> ObjectObservation | None:
        if detection.results_length <= 0:
            return None
        best = max(
            detection.results[: detection.results_length],
            key=lambda result: result.hypothesis.score,
        )
        label = str(best.hypothesis.class_id).strip()
        if not label:
            return None
        center = detection.bbox.center.position
        size = detection.bbox.size
        source_track_id, descriptor = decode_detection_id(str(detection.id))
        position = (float(center.x), float(center.y), float(center.z))
        anchor_keyframe = len(keyframe_poses) - 1 if keyframe_poses else None
        anchor_position = None
        if anchor_keyframe is not None:
            pose = keyframe_poses[anchor_keyframe]
            anchor_position_array = Rotation.from_quat(pose[3:7]).inv().apply(
                np.asarray(position) - np.asarray(pose[:3])
            )
            anchor_position = (
                float(anchor_position_array[0]),
                float(anchor_position_array[1]),
                float(anchor_position_array[2]),
            )
        return ObjectObservation(
            label=label,
            position=position,
            size=(float(size.x), float(size.y), float(size.z)),
            confidence=float(best.hypothesis.score),
            timestamp=to_timestamp(detection.header.stamp) or fallback_ts,
            source_track_id=source_track_id,
            descriptor=descriptor,
            anchor_keyframe=anchor_keyframe,
            anchor_position=anchor_position,
        )

    @staticmethod
    def _to_message(
        objects: list[SemanticObject],
        timestamp: float,
        frame_id: str,
    ) -> Detection3DArray:
        detections: list[Detection3D] = []
        for obj in objects:
            detection = Detection3D()
            detection.id = obj.object_id
            detection.header = Header(obj.last_seen, frame_id)
            detection.results = [
                ObjectHypothesisWithPose(
                    hypothesis=ObjectHypothesis(
                        class_id=obj.label,
                        score=obj.confidence,
                    )
                )
            ]
            detection.results_length = 1
            detection.bbox = BoundingBox3D(
                center=Pose(
                    position=Vector3(*obj.position),
                    orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
                ),
                size=Vector3(*obj.size),
            )
            detections.append(detection)
        return Detection3DArray(
            detections_length=len(detections),
            header=Header(timestamp, frame_id),
            detections=detections,
        )
