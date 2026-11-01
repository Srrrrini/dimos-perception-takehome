# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest

from dimos.perception.detection.detectors.grounding_dino import GroundingDino2DDetector
from dimos.perception.detection.detectors.yoloe import Yoloe2DDetector
from dimos.perception.detection.module3D import Detection3DModule
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2_semantic_map import (
    unitree_go2_semantic_map_temporal_grounding_dino,
    unitree_go2_semantic_map_temporal_yoloe,
)


@pytest.mark.parametrize(
    ("blueprint", "detector"),
    [
        (unitree_go2_semantic_map_temporal_yoloe, Yoloe2DDetector),
        (unitree_go2_semantic_map_temporal_grounding_dino, GroundingDino2DDetector),
    ],
)
def test_open_vocabulary_blueprint_configures_detector(blueprint, detector) -> None:  # type: ignore[no-untyped-def]
    detection = next(atom for atom in blueprint.blueprints if atom.module is Detection3DModule)

    assert detection.kwargs["detector"] is detector
