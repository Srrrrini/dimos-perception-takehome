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

import pytest

from dimos.perception.semantic_map.tracker import (
    GlobalAssignmentSemanticMapTracker,
    ObjectObservation,
    SemanticMapTracker,
)


def observation(
    label: str,
    x: float,
    timestamp: float,
    track_id: str = "",
) -> ObjectObservation:
    return ObjectObservation(
        label=label,
        position=(x, 2.0, 0.5),
        size=(0.5, 0.5, 1.0),
        confidence=0.8,
        timestamp=timestamp,
        source_track_id=track_id,
    )


def test_same_class_nearby_observations_are_fused() -> None:
    tracker = SemanticMapTracker(association_distance_m=0.5)

    first = tracker.update([observation("chair", 1.0, 1.0)])[0]
    second = tracker.update([observation("Chair", 1.2, 2.0)])[0]

    assert first is second
    assert second.observation_count == 2
    assert second.position == (1.1, 2.0, 0.5)
    assert second.first_seen == 1.0
    assert second.last_seen == 2.0
    assert second.position_stddev_m == pytest.approx(0.1414, abs=1e-4)


def test_different_labels_at_same_position_remain_distinct() -> None:
    tracker = SemanticMapTracker(association_distance_m=0.5)

    objects = tracker.update([observation("person", 1.0, 1.0), observation("chair", 1.0, 1.0)])

    assert len(objects) == 2
    assert len(tracker.objects) == 2


def test_two_same_class_observations_in_one_frame_cannot_collapse() -> None:
    tracker = SemanticMapTracker(association_distance_m=1.0)

    tracker.update([observation("chair", 0.0, 1.0), observation("chair", 0.2, 1.0)])

    assert len(tracker.objects) == 2


def test_far_observations_create_distinct_objects_and_query_confirmed_only() -> None:
    tracker = SemanticMapTracker(association_distance_m=0.5)
    tracker.update([observation("chair", 0.0, 1.0), observation("chair", 2.0, 1.0)])
    tracker.update([observation("chair", 0.1, 2.0)])

    assert len(tracker.find("chair", min_observations=2)) == 1
    assert tracker.find("air", min_observations=2)[0].observation_count == 2


def test_global_assignment_avoids_order_dependent_duplicate() -> None:
    tracker = GlobalAssignmentSemanticMapTracker(association_distance_m=0.35)
    tracker.update([observation("chair", 0.0, 1.0), observation("chair", 0.4, 1.0)])

    tracker.update([observation("chair", 0.3, 2.0), observation("chair", 0.6, 2.0)])

    assert len(tracker.objects) == 2
    assert [obj.observation_count for obj in tracker.objects] == [2, 2]


def test_global_assignment_uses_source_track_as_bounded_tiebreaker() -> None:
    tracker = GlobalAssignmentSemanticMapTracker(association_distance_m=1.0)
    tracker.update(
        [
            observation("chair", 0.0, 1.0, "left"),
            observation("chair", 0.4, 1.0, "right"),
        ]
    )

    updated = tracker.update(
        [
            observation("chair", 0.25, 2.0, "left"),
            observation("chair", 0.15, 2.0, "right"),
        ]
    )

    assert updated[0].source_track_id == "left"
    assert updated[1].source_track_id == "right"
