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
import math
import uuid


@dataclass(frozen=True)
class ObjectObservation:
    """One world-frame object observation."""

    label: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    confidence: float
    timestamp: float
    source_track_id: str = ""


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

    def update(self, observation: ObjectObservation) -> None:
        """Fuse an observation with an incremental arithmetic mean."""
        old_weight = self.observation_count
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
        self.observation_count = new_weight
        if observation.source_track_id:
            self.source_track_id = observation.source_track_id

    @property
    def position_stddev_m(self) -> float:
        """Root-mean-square 3D centroid deviation across observations."""
        if self.observation_count < 2:
            return 0.0
        return math.sqrt(self.position_m2 / (self.observation_count - 1))


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
                match.update(observation)
            updated.append(match)
        return updated

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
