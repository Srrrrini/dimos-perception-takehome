# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Deterministic component evaluation for temporal semantic-map association."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import pairwise

from dimos.perception.semantic_map.tracker import (
    ObjectObservation,
    TemporalSemanticMapTracker,
)


@dataclass(frozen=True)
class TrackingEvaluation:
    """Ground-truth association metrics for a controlled observation sequence."""

    ground_truth_objects: int
    observed_ground_truth_objects: int
    track_recall: float
    duplicate_track_count: int
    identity_switch_count: int
    false_merge_count: int
    moved_reidentifications: int
    moved_reidentification_accuracy: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _observation(
    label: str,
    position: tuple[float, float, float],
    timestamp: float,
    descriptor: tuple[float, ...],
) -> ObjectObservation:
    return ObjectObservation(
        label=label,
        position=position,
        size=(0.5, 0.5, 1.0),
        confidence=0.9,
        timestamp=timestamp,
        descriptor=descriptor,
    )


def evaluate_controlled_temporal_tracking() -> TrackingEvaluation:
    """Evaluate nearby lookalikes, occlusion/revisit, and one moved object.

    This is deliberately a tracker-level evaluation. It isolates association
    behavior from detector and depth errors; replay metrics cover the complete
    perception pipeline separately.
    """
    lookalike = (1.0, 0.0, 0.0)
    frames = (
        (
            0.0,
            (
                ("chair-1", "chair", (0.0, 0.0, 0.5), lookalike),
                ("chair-2", "chair", (1.5, 0.0, 0.5), lookalike),
                ("chair-3", "chair", (3.0, 0.0, 0.5), lookalike),
                ("chair-4", "chair", (4.5, 0.0, 0.5), lookalike),
                ("box-moved", "box", (7.0, 0.0, 0.5), (0.0, 1.0, 0.0)),
            ),
        ),
        (
            1.0,
            (
                ("chair-1", "chair", (0.05, 0.0, 0.5), lookalike),
                ("chair-2", "chair", (1.45, 0.0, 0.5), lookalike),
                ("chair-3", "chair", (3.05, 0.0, 0.5), lookalike),
                ("chair-4", "chair", (4.45, 0.0, 0.5), lookalike),
            ),
        ),
        (
            10.0,
            (
                # chair-2 was occluded; position recovers its original ID.
                ("chair-2", "chair", (1.52, 0.0, 0.5), lookalike),
                # The unique box descriptor recovers its ID after movement.
                ("box-moved", "box", (9.0, 0.0, 0.5), (0.0, 0.999, 0.001)),
            ),
        ),
    )
    tracker = TemporalSemanticMapTracker(
        association_distance_m=0.6,
        dormant_after_s=5.0,
        reidentification_distance_m=5.0,
    )
    assignments: dict[str, list[str]] = defaultdict(list)
    predicted_to_ground_truth: dict[str, set[str]] = defaultdict(set)
    for timestamp, ground_truth in frames:
        observations = [
            _observation(label, position, timestamp, descriptor)
            for _, label, position, descriptor in ground_truth
        ]
        predicted = tracker.update(observations)
        for (ground_truth_id, _, _, _), semantic_object in zip(
            ground_truth, predicted, strict=True
        ):
            assignments[ground_truth_id].append(semantic_object.object_id)
            predicted_to_ground_truth[semantic_object.object_id].add(ground_truth_id)

    all_ground_truth = {item[0] for _, frame in frames for item in frame}
    duplicate_tracks = sum(max(0, len(set(ids)) - 1) for ids in assignments.values())
    identity_switches = sum(
        sum(previous != current for previous, current in pairwise(ids))
        for ids in assignments.values()
    )
    false_merges = sum(len(ids) - 1 for ids in predicted_to_ground_truth.values() if len(ids) > 1)
    moved_ids = assignments["box-moved"]
    moved_reidentifications = int(len(moved_ids) == 2 and moved_ids[0] == moved_ids[1])
    observed = len(assignments)
    return TrackingEvaluation(
        ground_truth_objects=len(all_ground_truth),
        observed_ground_truth_objects=observed,
        track_recall=observed / len(all_ground_truth),
        duplicate_track_count=duplicate_tracks,
        identity_switch_count=identity_switches,
        false_merge_count=false_merges,
        moved_reidentifications=moved_reidentifications,
        moved_reidentification_accuracy=float(moved_reidentifications),
    )
