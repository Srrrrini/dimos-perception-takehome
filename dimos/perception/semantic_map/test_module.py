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

from collections.abc import Iterator
from unittest.mock import MagicMock

from dimos_lcm.vision_msgs import BoundingBox3D, ObjectHypothesis, ObjectHypothesisWithPose
import pytest

from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.std_msgs.Header import Header
from dimos.msgs.vision_msgs.Detection3D import Detection3D
from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray
from dimos.perception.semantic_map.module import SemanticObjectMap
from dimos.perception.semantic_map.tracker import SemanticMapTracker, TemporalSemanticMapTracker


def detection(label: str, x: float, timestamp: float) -> Detection3D:
    msg = Detection3D()
    msg.header = Header(timestamp, "world")
    msg.results = [
        ObjectHypothesisWithPose(
            hypothesis=ObjectHypothesis(class_id=label, score=0.9),
        )
    ]
    msg.results_length = 1
    msg.bbox = BoundingBox3D(
        center=Pose(position=Vector3(x, 2.0, 0.5)),
        size=Vector3(0.5, 0.5, 1.0),
    )
    return msg


def frame(*detections: Detection3D, timestamp: float) -> Detection3DArray:
    return Detection3DArray(
        detections_length=len(detections),
        header=Header(timestamp, "world"),
        detections=list(detections),
    )


@pytest.fixture
def semantic_map() -> Iterator[SemanticObjectMap]:
    module = SemanticObjectMap(min_observations=2, association_distance_m=0.5)
    module.semantic_map = MagicMock()
    yield module
    module.stop()


def test_module_confirms_publishes_and_queries_persistent_object(
    semantic_map: SemanticObjectMap,
) -> None:
    semantic_map._on_detections(frame(detection("chair", 1.0, 1.0), timestamp=1.0))
    semantic_map._on_detections(frame(detection("chair", 1.2, 2.0), timestamp=2.0))

    published = semantic_map.semantic_map.publish.call_args.args[0]
    assert published.to_json() == [
        {
            "label": "chair id=" + published.detections[0].id,
            "score": 0.9,
            "position": {"x": 1.1, "y": 2.0, "z": 0.5},
            "size": {"x": 0.5, "y": 0.5, "z": 1.0},
        }
    ]
    assert "chair" in semantic_map.where_is_object("chair")
    assert "(1.10, 2.00, 0.50) m in world" in semantic_map.where_is_object("chair")
    metrics = semantic_map.get_metrics()
    assert metrics["confirmed_objects"] == 1
    assert metrics["frames_3d"] == 2
    assert metrics["observations_in"] == 2
    assert metrics["matched_observations"] == 1
    assert metrics["new_tracks"] == 1
    assert metrics["association_rate"] == 0.5
    assert metrics["same_label_pairs_within_1_0m"] == 0


def test_unconfirmed_object_is_not_returned(semantic_map: SemanticObjectMap) -> None:
    message = frame(detection("plant", 1.0, 1.0), timestamp=1.0)
    transported = Detection3DArray.lcm_decode(message.lcm_encode())
    semantic_map._on_detections(transported)

    assert semantic_map.where_is_object("plant").startswith("No confirmed object")
    assert semantic_map.get_semantic_map().detections_length == 0


def test_module_accepts_injected_map_backend() -> None:
    backend = SemanticMapTracker(association_distance_m=0.25)
    module = SemanticObjectMap(
        association_distance_m=0.5,
        backend=lambda _distance: backend,
    )

    assert module._tracker is backend
    assert module.get_metrics()["backend"] == "SemanticMapTracker"

    module.stop()


def test_pending_move_is_not_counted_as_an_accepted_fusion() -> None:
    module = SemanticObjectMap(
        min_observations=1,
        association_distance_m=0.5,
        backend=lambda distance: TemporalSemanticMapTracker(
            association_distance_m=distance,
            allow_source_track_relocation=True,
        ),
    )
    module.semantic_map = MagicMock()
    first = detection("chair", 1.0, 1.0)
    first.id = "persistent"
    proposal = detection("chair", 3.0, 2.0)
    proposal.id = "persistent"

    module._on_detections(frame(first, timestamp=1.0))
    module._on_detections(frame(proposal, timestamp=2.0))

    metrics = module.get_metrics()
    assert metrics["accepted_fusions"] == 0
    assert metrics["unfused_observations"] == 1
    assert metrics["association_rate"] == 0.0
    module.stop()


def test_invalid_frame_and_nonfinite_detection_are_rejected() -> None:
    module = SemanticObjectMap(min_observations=1)
    module.semantic_map = MagicMock()
    wrong_frame = frame(detection("chair", 1.0, 1.0), timestamp=1.0)
    wrong_frame.header.frame_id = "camera"
    invalid = detection("chair", float("nan"), 2.0)

    module._on_detections(wrong_frame)
    module._on_detections(frame(invalid, timestamp=2.0))

    metrics = module.get_metrics()
    assert metrics["rejected_frames"] == 1
    assert metrics["invalid_observations"] == 1
    assert metrics["candidate_objects"] == 0
    module.stop()


def test_pose_graph_correction_is_disabled_by_default() -> None:
    module = SemanticObjectMap(min_observations=1)

    assert module.get_metrics()["pose_graph_correction_enabled"] is False
    assert module.get_metrics()["graph_corrections"] == 0
    module.stop()
