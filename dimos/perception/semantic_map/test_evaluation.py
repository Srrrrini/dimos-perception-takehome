# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from dimos.perception.semantic_map.evaluation import evaluate_controlled_temporal_tracking


def test_controlled_temporal_tracking_evaluation() -> None:
    metrics = evaluate_controlled_temporal_tracking()

    assert metrics.ground_truth_objects == 5
    assert metrics.track_recall == 1.0
    assert metrics.duplicate_track_count == 0
    assert metrics.identity_switch_count == 0
    assert metrics.false_merge_count == 0
    assert metrics.moved_reidentification_accuracy == 1.0
