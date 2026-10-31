#!/usr/bin/env python3
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

"""Go2 replay blueprint for persistent 3D semantic object mapping."""

from dimos.core.coordination.blueprints import autoconnect
from dimos.perception.detection.detectors.yolo import Yolo11Seg2DDetector
from dimos.perception.detection.module3D import Detection3DModule
from dimos.perception.semantic_map.module import SemanticObjectMap
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2 import unitree_go2
from dimos.robot.unitree.go2.connection import GO2Connection

unitree_go2_semantic_map = (
    autoconnect(
        unitree_go2,
        Detection3DModule.blueprint(camera_info=GO2Connection.camera_info_static),
        SemanticObjectMap.blueprint(),
    )
    .remappings([(Detection3DModule, "pointcloud", "global_map")])
    .global_config(n_workers=12, robot_model="unitree_go2")
)

unitree_go2_semantic_map_seg = (
    autoconnect(
        unitree_go2,
        Detection3DModule.blueprint(
            camera_info=GO2Connection.camera_info_static,
            detector=Yolo11Seg2DDetector,
        ),
        SemanticObjectMap.blueprint(),
    )
    .remappings([(Detection3DModule, "pointcloud", "global_map")])
    .global_config(n_workers=12, robot_model="unitree_go2")
)
