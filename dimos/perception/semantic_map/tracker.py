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

"""Small, deterministic association core for a semantic object map."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import uuid


class TrackState(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    MOVED = "moved"


@dataclass(frozen=True)
class ObjectObservation:
    """One world-frame object observation."""

    label: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    confidence: float
    timestamp: float
    source_track_id: str = ""
    descriptor: tuple[float, ...] = ()


@dataclass
class SemanticObject:
    """Persistent estimate made from repeated object observations."""

    object_id: str
    label: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    confidence: float
    first_seen: float
    last_seen: float
    observation_count: int = 1
    source_track_id: str = ""
    position_m2: float = 0.0
    position_sample_count: int = 1
    state: TrackState = TrackState.ACTIVE
    move_count: int = 0
    descriptor: tuple[float, ...] = ()

    def update(self, observation: ObjectObservation) -> None:
        """Fuse an observation with an incremental arithmetic mean."""
        old_weight = self.position_sample_count
        new_weight = old_weight + 1
        previous_position = self.position
        updated_position = (
            (old_weight * self.position[0] + observation.position[0]) / new_weight,
            (old_weight * self.position[1] + observation.position[1]) / new_weight,
            (old_weight * self.position[2] + observation.position[2]) / new_weight,
        )
        self.position_m2 += sum(
            (current - previous) * (current - updated)
            for previous, current, updated in zip(
                previous_position,
                observation.position,
                updated_position,
                strict=True,
            )
        )
        self.position = updated_position
        self.size = (
            (old_weight * self.size[0] + observation.size[0]) / new_weight,
            (old_weight * self.size[1] + observation.size[1]) / new_weight,
            (old_weight * self.size[2] + observation.size[2]) / new_weight,
        )
        self.confidence = max(self.confidence, observation.confidence)
        self.last_seen = max(self.last_seen, observation.timestamp)
        self.observation_count += 1
        self.position_sample_count = new_weight
        if self.state != TrackState.MOVED:
            self.state = TrackState.ACTIVE
        if observation.source_track_id:
            self.source_track_id = observation.source_track_id
        if observation.descriptor:
            self.descriptor = observation.descriptor

    def relocate(self, observation: ObjectObservation) -> None:
        """Preserve identity while starting a new position estimate after movement."""
        self.position = observation.position
        self.size = observation.size
        self.confidence = max(self.confidence, observation.confidence)
        self.last_seen = max(self.last_seen, observation.timestamp)
        self.observation_count += 1
        self.position_m2 = 0.0
        self.position_sample_count = 1
        self.state = TrackState.MOVED
        self.move_count += 1
        if observation.source_track_id:
            self.source_track_id = observation.source_track_id
        if observation.descriptor:
            self.descriptor = observation.descriptor

    @property
    def position_stddev_m(self) -> float:
        """Root-mean-square 3D centroid deviation across observations."""
        if self.position_sample_count < 2:
            return 0.0
        return math.sqrt(self.position_m2 / (self.position_sample_count - 1))


class SemanticMapTracker:
    """Associate same-label observations by stable world position."""

    def __init__(self, association_distance_m: float = 0.75) -> None:
        if association_distance_m <= 0:
            raise ValueError("association_distance_m must be positive")
        self.association_distance_m = association_distance_m
        self._objects: dict[str, SemanticObject] = {}

    @property
    def objects(self) -> tuple[SemanticObject, ...]:
        return tuple(self._objects.values())

    def update(self, observations: list[ObjectObservation]) -> list[SemanticObject]:
        """Associate one frame without assigning two observations to one object."""
        updated: list[SemanticObject] = []
        claimed: set[str] = set()
        for observation in observations:
            match = self._nearest_match(observation, claimed)
            if match is None:
                match = self._new_object(observation)
            else:
                match.update(observation)
            claimed.add(match.object_id)
            updated.append(match)
        return updated

    def _new_object(self, observation: ObjectObservation) -> SemanticObject:
        obj = SemanticObject(
            object_id=uuid.uuid4().hex,
            label=observation.label,
            position=observation.position,
            size=observation.size,
            confidence=observation.confidence,
            first_seen=observation.timestamp,
            last_seen=observation.timestamp,
            source_track_id=observation.source_track_id,
            descriptor=observation.descriptor,
        )
        self._objects[obj.object_id] = obj
        return obj

    def confirmed(self, min_observations: int) -> list[SemanticObject]:
        """Return stable objects in deterministic insertion order."""
        return [obj for obj in self._objects.values() if obj.observation_count >= min_observations]

    def find(self, query: str, min_observations: int) -> list[SemanticObject]:
        """Find confirmed objects by case-insensitive exact or substring label."""
        normalized = query.strip().casefold()
        if not normalized:
            return []
        candidates = self.confirmed(min_observations)
        exact = [obj for obj in candidates if obj.label.casefold() == normalized]
        if exact:
            return exact
        return [obj for obj in candidates if normalized in obj.label.casefold()]

    def _nearest_match(
        self,
        observation: ObjectObservation,
        claimed: set[str],
    ) -> SemanticObject | None:
        candidates = [
            obj
            for obj in self._objects.values()
            if obj.object_id not in claimed and obj.label.casefold() == observation.label.casefold()
        ]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda obj: math.dist(obj.position, observation.position))
        if math.dist(nearest.position, observation.position) > self.association_distance_m:
            return None
        return nearest


class GlobalAssignmentSemanticMapTracker(SemanticMapTracker):
    """Globally minimize gated same-label association cost for each frame."""

    def update(self, observations: list[ObjectObservation]) -> list[SemanticObject]:
        if not observations:
            return []

        from scipy.optimize import linear_sum_assignment

        existing = list(self._objects.values())
        if not existing:
            return [self._new_object(observation) for observation in observations]

        invalid_cost = 1_000_000.0
        costs = [
            [self._association_cost(observation, obj, invalid_cost) for obj in existing]
            for observation in observations
        ]
        rows, columns = linear_sum_assignment(costs)
        assignments = {
            int(row): existing[int(column)]
            for row, column in zip(rows, columns, strict=True)
            if costs[int(row)][int(column)] < invalid_cost
        }

        updated: list[SemanticObject] = []
        for index, observation in enumerate(observations):
            match = assignments.get(index)
            if match is None:
                match = self._new_object(observation)
            else:
                self._update_match(match, observation)
            updated.append(match)
        return updated

    def _update_match(self, match: SemanticObject, observation: ObjectObservation) -> None:
        match.update(observation)

    def _association_cost(
        self,
        observation: ObjectObservation,
        obj: SemanticObject,
        invalid_cost: float,
    ) -> float:
        if obj.label.casefold() != observation.label.casefold():
            return invalid_cost
        distance = math.dist(obj.position, observation.position)
        if distance > self.association_distance_m:
            return invalid_cost
        if observation.source_track_id and observation.source_track_id == obj.source_track_id:
            distance -= min(0.25, self.association_distance_m * 0.25)
        return distance


class TemporalSemanticMapTracker(GlobalAssignmentSemanticMapTracker):
    """Global assignment with dormant tracks and conservative moved-object re-identification."""

    def __init__(
        self,
        association_distance_m: float = 0.75,
        dormant_after_s: float = 5.0,
        reidentification_distance_m: float = 5.0,
        descriptor_similarity: float = 0.85,
        source_track_reid_max_age_s: float = 2.0,
        move_confirmation_observations: int = 3,
    ) -> None:
        super().__init__(association_distance_m)
        if dormant_after_s <= 0:
            raise ValueError("dormant_after_s must be positive")
        if reidentification_distance_m <= association_distance_m:
            raise ValueError("reidentification_distance_m must exceed the association gate")
        self.dormant_after_s = dormant_after_s
        self.reidentification_distance_m = reidentification_distance_m
        if not 0.0 < descriptor_similarity <= 1.0:
            raise ValueError("descriptor_similarity must be in (0, 1]")
        self.descriptor_similarity = descriptor_similarity
        if source_track_reid_max_age_s <= 0:
            raise ValueError("source_track_reid_max_age_s must be positive")
        self.source_track_reid_max_age_s = source_track_reid_max_age_s
        if move_confirmation_observations <= 1:
            raise ValueError("move_confirmation_observations must exceed one")
        self.move_confirmation_observations = move_confirmation_observations
        self._pending_moves: dict[str, tuple[tuple[float, float, float], int, float]] = {}

    def update(self, observations: list[ObjectObservation]) -> list[SemanticObject]:
        if observations:
            now = max(observation.timestamp for observation in observations)
            for obj in self._objects.values():
                if now - obj.last_seen >= self.dormant_after_s and obj.state == TrackState.ACTIVE:
                    obj.state = TrackState.DORMANT
        return super().update(observations)

    def _association_cost(
        self,
        observation: ObjectObservation,
        obj: SemanticObject,
        invalid_cost: float,
    ) -> float:
        normal_cost = super()._association_cost(observation, obj, invalid_cost)
        if normal_cost < invalid_cost:
            return normal_cost
        same_source_identity = (
            observation.source_track_id
            and observation.source_track_id == obj.source_track_id
            and observation.timestamp - obj.last_seen <= self.source_track_reid_max_age_s
        )
        descriptor_match = self._descriptor_similarity(observation.descriptor, obj.descriptor)
        if observation.label.casefold() == obj.label.casefold() and (
            same_source_identity or descriptor_match >= self.descriptor_similarity
        ):
            distance = math.dist(obj.position, observation.position)
            if distance <= self.reidentification_distance_m:
                return self.association_distance_m + distance - descriptor_match * 0.1
        return invalid_cost

    def _update_match(self, match: SemanticObject, observation: ObjectObservation) -> None:
        if math.dist(match.position, observation.position) <= self.association_distance_m:
            self._pending_moves.pop(match.object_id, None)
            match.update(observation)
            return

        descriptor_match = self._descriptor_similarity(observation.descriptor, match.descriptor)
        if descriptor_match >= self.descriptor_similarity:
            self._pending_moves.pop(match.object_id, None)
            match.relocate(observation)
            return

        previous = self._pending_moves.get(match.object_id)
        if (
            previous is None
            or observation.timestamp - previous[2] > self.source_track_reid_max_age_s
            or math.dist(previous[0], observation.position) > self.association_distance_m
        ):
            count = 1
            mean_position = observation.position
        else:
            count = previous[1] + 1
            mean_position = (
                (previous[0][0] * previous[1] + observation.position[0]) / count,
                (previous[0][1] * previous[1] + observation.position[1]) / count,
                (previous[0][2] * previous[1] + observation.position[2]) / count,
            )
        self._pending_moves[match.object_id] = (mean_position, count, observation.timestamp)
        if count >= self.move_confirmation_observations:
            match.relocate(observation)
            self._pending_moves.pop(match.object_id, None)

    @staticmethod
    def _descriptor_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        if not left or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)
