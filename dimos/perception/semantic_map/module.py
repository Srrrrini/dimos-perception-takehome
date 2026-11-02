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

from collections import Counter
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
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
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
    expected_frame_id: str = "world"
    max_timestamp_lateness_s: float = 0.25
    enable_experimental_pose_graph_correction: bool = False
    navigation_standoff_m: float = 1.0


class SemanticObjectMap(Module):
    """Build and query a persistent 3D semantic object map."""

    config: SemanticObjectMapConfig

    detections_3d: In[Detection3DArray]
    pgo_keyframes: In[PoseArray]
    odom: In[PoseStamped]
    semantic_map: Out[Detection3DArray]
    goal_request: Out[PoseStamped]

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.config.min_observations <= 0:
            raise ValueError("min_observations must be positive")
        if not self.config.expected_frame_id:
            raise ValueError("expected_frame_id must not be empty")
        if self.config.max_timestamp_lateness_s < 0:
            raise ValueError("max_timestamp_lateness_s must be non-negative")
        if self.config.navigation_standoff_m <= 0:
            raise ValueError("navigation_standoff_m must be positive")
        self._tracker = self.config.backend(self.config.association_distance_m)
        self._lock = RLock()
        self._metrics_started_at = time.perf_counter()
        self._frames_3d = 0
        self._observations_in = 0
        self._matched_observations = 0
        self._unfused_observations = 0
        self._new_tracks = 0
        self._invalid_observations = 0
        self._rejected_frames = 0
        self._association_ms: list[float] = []
        self._keyframe_poses: tuple[tuple[float, ...], ...] = ()
        self._latest_detection_ts = -math.inf
        self._latest_pose: PoseStamped | None = None
        self._navigation_requests = 0
        self._navigation_failures = 0

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.detections_3d.subscribe(self._on_detections)))
        self.register_disposable(Disposable(self.pgo_keyframes.subscribe(self._on_keyframes)))
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))

    def _on_odom(self, pose: PoseStamped) -> None:
        values = (*pose.position.to_tuple(), *pose.orientation.to_tuple(), float(pose.ts))
        if pose.frame_id == self.config.expected_frame_id and all(map(math.isfinite, values)):
            with self._lock:
                self._latest_pose = pose

    def _on_detections(self, message: Detection3DArray) -> None:
        started_at = time.perf_counter()
        message_ts = float(message.ts)
        with self._lock:
            invalid_frame = (
                message.frame_id != self.config.expected_frame_id
                or not math.isfinite(message_ts)
                or message_ts < self._latest_detection_ts - self.config.max_timestamp_lateness_s
            )
            if invalid_frame:
                self._rejected_frames += 1
            else:
                self._latest_detection_ts = max(self._latest_detection_ts, message_ts)
        if invalid_frame:
            logger.warning(
                "Rejected semantic-map frame",
                frame_id=message.frame_id,
                timestamp=message_ts,
            )
            return
        with self._lock:
            keyframe_poses = (
                self._keyframe_poses
                if self.config.enable_experimental_pose_graph_correction
                else ()
            )
        observations = []
        invalid = 0
        for detection in message.detections[: message.detections_length]:
            observation = self._to_observation(
                detection,
                message.ts,
                keyframe_poses,
                expected_frame_id=self.config.expected_frame_id,
            )
            if observation is None:
                invalid += 1
            else:
                observations.append(observation)
        with self._lock:
            self._frames_3d += 1
            self._observations_in += len(observations)
            self._invalid_observations += invalid
        if not observations:
            return
        with self._lock:
            objects_before = len(self._tracker.objects)
            samples_before = sum(obj.observation_count for obj in self._tracker.objects)
            self._tracker.update(observations)
            created = len(self._tracker.objects) - objects_before
            samples_after = sum(obj.observation_count for obj in self._tracker.objects)
            accepted_fusions = max(0, samples_after - samples_before - created)
            self._new_tracks += created
            self._matched_observations += accepted_fusions
            self._unfused_observations += max(0, len(observations) - created - accepted_fusions)
            self._association_ms.append((time.perf_counter() - started_at) * 1000.0)
            mapped = self._tracker.confirmed(self.config.min_observations)
        self.semantic_map.publish(self._to_message(mapped, message.ts, message.frame_id))

    def _on_keyframes(self, message: PoseArray) -> None:
        if not self.config.enable_experimental_pose_graph_correction:
            return
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
        if message.frame_id != self.config.expected_frame_id or not all(
            all(math.isfinite(value) for value in pose)
            and math.isclose(sum(value * value for value in pose[3:7]), 1.0, abs_tol=1e-3)
            for pose in poses
        ):
            with self._lock:
                self._rejected_frames += 1
            logger.warning("Rejected invalid pose-graph update", frame_id=message.frame_id)
            return
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

    @skill
    def navigate_to_object(self, name: str) -> str:
        """Send the existing planner a standoff goal near the nearest matching object."""
        with self._lock:
            matches = self._tracker.find(name, self.config.min_observations)
            pose = self._latest_pose
            if not matches:
                self._navigation_failures += 1
                return f"No confirmed object matching '{name}' is in the semantic map."
            if pose is None:
                self._navigation_failures += 1
                return "Cannot navigate without a valid world-frame robot pose."
            target = min(
                matches,
                key=lambda obj: math.hypot(
                    obj.position[0] - pose.position.x,
                    obj.position[1] - pose.position.y,
                ),
            )
            dx = target.position[0] - pose.position.x
            dy = target.position[1] - pose.position.y
            distance = math.hypot(dx, dy)
            travel = max(0.0, distance - self.config.navigation_standoff_m)
            scale = travel / distance if distance > 1e-6 else 0.0
            goal_x = pose.position.x + dx * scale
            goal_y = pose.position.y + dy * scale
            yaw = math.atan2(dy, dx) if distance > 1e-6 else 0.0
            goal = PoseStamped(
                goal_x,
                goal_y,
                pose.position.z,
                0.0,
                0.0,
                math.sin(yaw / 2.0),
                math.cos(yaw / 2.0),
                ts=time.time(),
                frame_id=self.config.expected_frame_id,
            )
            self._navigation_requests += 1
        self.goal_request.publish(goal)
        return (
            f"Navigation goal sent for {target.label} {target.object_id[:8]}: "
            f"({goal_x:.2f}, {goal_y:.2f}) m, "
            f"standoff={self.config.navigation_standoff_m:.2f} m."
        )

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
            unfused_observations = self._unfused_observations
            invalid_observations = self._invalid_observations
            rejected_frames = self._rejected_frames
            association_ms = list(self._association_ms)
            navigation_requests = self._navigation_requests
            navigation_failures = self._navigation_failures
        observations = sum(obj.observation_count for obj in confirmed)
        deviations = [obj.position_stddev_m for obj in confirmed]
        lifetimes = [obj.last_seen - obj.first_seen for obj in confirmed]
        candidate_labels = Counter(obj.label for obj in all_objects)
        confirmed_labels = Counter(obj.label for obj in confirmed)
        return {
            "candidate_objects": len(all_objects),
            "confirmed_objects": len(confirmed),
            "candidate_objects_by_label": dict(sorted(candidate_labels.items())),
            "confirmed_objects_by_label": dict(sorted(confirmed_labels.items())),
            "confirmation_ratio": len(confirmed) / len(all_objects) if all_objects else 0.0,
            "frames_3d": frames_3d,
            "observations_in": observations_in,
            "matched_observations": matched_observations,
            "accepted_fusions": matched_observations,
            "unfused_observations": unfused_observations,
            "invalid_observations": invalid_observations,
            "rejected_frames": rejected_frames,
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
            "pose_graph_correction_enabled": (
                self.config.enable_experimental_pose_graph_correction
            ),
            "navigation_requests": navigation_requests,
            "navigation_failures": navigation_failures,
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
        expected_frame_id: str = "world",
    ) -> ObjectObservation | None:
        if detection.results_length <= 0 or detection.header.frame_id != expected_frame_id:
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
        dimensions = (float(size.x), float(size.y), float(size.z))
        confidence = float(best.hypothesis.score)
        timestamp = to_timestamp(detection.header.stamp) or fallback_ts
        if (
            not all(
                math.isfinite(value) for value in (*position, *dimensions, confidence, timestamp)
            )
            or any(value <= 0.0 for value in dimensions)
            or not 0.0 <= confidence <= 1.0
        ):
            return None
        anchor_keyframe = len(keyframe_poses) - 1 if keyframe_poses else None
        anchor_position = None
        if anchor_keyframe is not None:
            pose = keyframe_poses[anchor_keyframe]
            anchor_position_array = (
                Rotation.from_quat(pose[3:7])
                .inv()
                .apply(np.asarray(position) - np.asarray(pose[:3]))
            )
            anchor_position = (
                float(anchor_position_array[0]),
                float(anchor_position_array[1]),
                float(anchor_position_array[2]),
            )
        return ObjectObservation(
            label=label,
            position=position,
            size=dimensions,
            confidence=confidence,
            timestamp=timestamp,
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
