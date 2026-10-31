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

"""Structural contract for replaceable semantic-map association backends."""

from typing import Protocol

from dimos.perception.semantic_map.tracker import ObjectObservation, SemanticObject


class SemanticMapBackend(Protocol):
    """Association, track fusion, and map lifecycle behind the DimOS module boundary."""

    @property
    def objects(self) -> tuple[SemanticObject, ...]: ...

    def update(self, observations: list[ObjectObservation]) -> list[SemanticObject]: ...

    def confirmed(self, min_observations: int) -> list[SemanticObject]: ...

    def find(self, query: str, min_observations: int) -> list[SemanticObject]: ...
