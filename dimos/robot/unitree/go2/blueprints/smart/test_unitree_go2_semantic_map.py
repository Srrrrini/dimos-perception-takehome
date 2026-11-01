# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest

from dimos.perception.detection.detectors.grounding_dino import (
    GroundingDino2DDetector,
    GroundingDinoSam2DDetector,
)
from dimos.perception.detection.detectors.yoloe import Yoloe2DDetector
from dimos.perception.detection.module3D import Detection3DModule
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2_semantic_map import (
    unitree_go2_semantic_map_temporal_grounding_dino,
    unitree_go2_semantic_map_temporal_grounding_dino_sam,
    unitree_go2_semantic_map_temporal_yoloe,
)


@pytest.mark.parametrize(
    ("blueprint", "detector"),
    [
        (unitree_go2_semantic_map_temporal_yoloe, Yoloe2DDetector),
        (unitree_go2_semantic_map_temporal_grounding_dino, GroundingDino2DDetector),
        (unitree_go2_semantic_map_temporal_grounding_dino_sam, GroundingDinoSam2DDetector),
    ],
)
def test_open_vocabulary_blueprint_configures_detector(blueprint, detector) -> None:  # type: ignore[no-untyped-def]
    detection = next(atom for atom in blueprint.blueprints if atom.module is Detection3DModule)

    assert detection.kwargs["detector"] is detector


def test_grounding_dino_blueprint_exposes_queries_as_configuration() -> None:
    detection = next(
        atom
        for atom in unitree_go2_semantic_map_temporal_grounding_dino.blueprints
        if atom.module is Detection3DModule
    )

    assert detection.kwargs["detector_kwargs"] == {
        "queries": GroundingDino2DDetector.DEFAULT_QUERIES
    }
