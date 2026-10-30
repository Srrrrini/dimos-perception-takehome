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

from threading import RLock

from dimos_lcm.vision_msgs import BoundingBox3D, ObjectHypothesis, ObjectHypothesisWithPose
from reactivex.disposable import Disposable

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.std_msgs.Header import Header
from dimos.msgs.vision_msgs.Detection3D import Detection3D
from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray
from dimos.perception.semantic_map.tracker import (
    ObjectObservation,
    SemanticMapTracker,
    SemanticObject,
)
from dimos.types.timestamped import to_timestamp


class SemanticObjectMapConfig(ModuleConfig):
    """Configuration for conservative world-space object association."""

    association_distance_m: float = 0.75
    min_observations: int = 2


class SemanticObjectMap(Module):
    """Build and query a persistent 3D semantic object map."""

    config: SemanticObjectMapConfig

    detections_3d: In[Detection3DArray]
    semantic_map: Out[Detection3DArray]

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.config.min_observations <= 0:
            raise ValueError("min_observations must be positive")
        self._tracker = SemanticMapTracker(self.config.association_distance_m)
        self._lock = RLock()

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.detections_3d.subscribe(self._on_detections)))

    def _on_detections(self, message: Detection3DArray) -> None:
        observations = [
            observation
            for detection in message.detections[: message.detections_length]
            if (observation := self._to_observation(detection, message.ts)) is not None
        ]
        if not observations:
            return
        with self._lock:
            self._tracker.update(observations)
            mapped = self._tracker.confirmed(self.config.min_observations)
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
    def get_metrics(self) -> dict[str, int | float]:
        """Return replay-friendly map persistence and consistency metrics."""
        with self._lock:
            all_objects = list(self._tracker.objects)
            confirmed = self._tracker.confirmed(self.config.min_observations)
        observations = sum(obj.observation_count for obj in confirmed)
        deviations = [obj.position_stddev_m for obj in confirmed]
        lifetimes = [obj.last_seen - obj.first_seen for obj in confirmed]
        return {
            "candidate_objects": len(all_objects),
            "confirmed_objects": len(confirmed),
            "confirmed_observations": observations,
            "mean_observations_per_object": observations / len(confirmed) if confirmed else 0.0,
            "mean_position_stddev_m": sum(deviations) / len(deviations) if deviations else 0.0,
            "max_position_stddev_m": max(deviations, default=0.0),
            "max_observed_track_duration_s": max(lifetimes, default=0.0),
        }

    @staticmethod
    def _to_observation(detection: Detection3D, fallback_ts: float) -> ObjectObservation | None:
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
        return ObjectObservation(
            label=label,
            position=(float(center.x), float(center.y), float(center.z)),
            size=(float(size.x), float(size.y), float(size.z)),
            confidence=float(best.hypothesis.score),
            timestamp=to_timestamp(detection.header.stamp) or fallback_ts,
            source_track_id=str(detection.id),
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
