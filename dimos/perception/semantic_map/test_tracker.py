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
    TemporalSemanticMapTracker,
    TrackState,
)


def observation(
    label: str,
    x: float,
    timestamp: float,
    track_id: str = "",
    descriptor: tuple[float, ...] = (),
) -> ObjectObservation:
    return ObjectObservation(
        label=label,
        position=(x, 2.0, 0.5),
        size=(0.5, 0.5, 1.0),
        confidence=0.8,
        timestamp=timestamp,
        source_track_id=track_id,
        descriptor=descriptor,
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


def test_temporal_tracker_revisits_stationary_object_without_new_identity() -> None:
    tracker = TemporalSemanticMapTracker(association_distance_m=0.5, dormant_after_s=5.0)
    original = tracker.update([observation("chair", 1.0, 1.0, "old")])[0]

    revisited = tracker.update([observation("chair", 1.1, 20.0, "new")])[0]

    assert revisited is original
    assert len(tracker.objects) == 1
    assert revisited.state == TrackState.ACTIVE


def test_temporal_tracker_preserves_identity_and_marks_moved_object() -> None:
    tracker = TemporalSemanticMapTracker(association_distance_m=0.5, dormant_after_s=5.0)
    original = tracker.update([observation("chair", 1.0, 1.0, "persistent")])[0]

    first_proposal = tracker.update([observation("chair", 3.0, 2.0, "persistent")])[0]
    assert first_proposal.state == TrackState.ACTIVE
    assert first_proposal.move_count == 0
    tracker.update([observation("chair", 3.1, 2.5, "persistent")])
    moved = tracker.update([observation("chair", 3.0, 3.0, "persistent")])[0]

    assert moved is original
    assert len(tracker.objects) == 1
    assert moved.position == (3.0, 2.0, 0.5)
    assert moved.state == TrackState.MOVED
    assert moved.move_count == 1


def test_temporal_tracker_rejects_single_frame_move_jump() -> None:
    tracker = TemporalSemanticMapTracker(association_distance_m=0.5, dormant_after_s=5.0)
    original = tracker.update([observation("chair", 1.0, 1.0, "persistent")])[0]

    tracker.update([observation("chair", 3.0, 2.0, "persistent")])

    assert original.position == (1.0, 2.0, 0.5)
    assert original.state == TrackState.ACTIVE
    assert original.move_count == 0


def test_temporal_tracker_does_not_guess_ambiguous_moved_identity() -> None:
    tracker = TemporalSemanticMapTracker(association_distance_m=0.5, dormant_after_s=5.0)
    tracker.update([observation("chair", 1.0, 1.0, "old")])

    tracker.update([observation("chair", 3.0, 10.0, "new")])

    assert len(tracker.objects) == 2


def test_temporal_tracker_reidentifies_moved_object_from_descriptor() -> None:
    tracker = TemporalSemanticMapTracker(association_distance_m=0.5, dormant_after_s=5.0)
    original = tracker.update(
        [observation("chair", 1.0, 1.0, "old", descriptor=(1.0, 0.0))]
    )[0]

    moved = tracker.update(
        [observation("chair", 3.0, 10.0, "new", descriptor=(0.99, 0.01))]
    )[0]

    assert moved is original
    assert moved.state == TrackState.MOVED
