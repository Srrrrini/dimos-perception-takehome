# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Replaceable appearance descriptors and compact Detection3D ID transport."""

from __future__ import annotations

from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Sequence
from typing import Protocol

import cv2
import numpy as np
from PIL import Image as PILImage

from dimos.perception.detection.type.detection2d.bbox import Detection2DBBox

_PREFIX = "dimos-reid-v1:"


class AppearanceDescriptor(Protocol):
    """Extract a unit-normalized descriptor from one image detection."""

    def describe(self, detection: Detection2DBBox) -> tuple[float, ...]: ...

    def describe_many(self, detections: Sequence[Detection2DBBox]) -> list[tuple[float, ...]]: ...


class NoAppearanceDescriptor:
    """Zero-cost strategy used by blueprints that do not request ReID."""

    def describe(self, detection: Detection2DBBox) -> tuple[float, ...]:
        return ()

    def describe_many(self, detections: Sequence[Detection2DBBox]) -> list[tuple[float, ...]]:
        return [() for _ in detections]


class HSVHistogramDescriptor:
    """Small offline baseline: normalized H/S/V histograms over the instance crop."""

    def describe(self, detection: Detection2DBBox) -> tuple[float, ...]:
        image = detection.image.to_opencv()
        height, width = image.shape[:2]
        x1, y1, x2, y2 = detection.bbox
        left, top = max(0, int(x1)), max(0, int(y1))
        right, bottom = min(width, int(x2)), min(height, int(y2))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            return ()
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        instance_mask = getattr(detection, "mask", None)
        mask = None if instance_mask is None else instance_mask[top:bottom, left:right]
        histograms = [
            cv2.calcHist([hsv], [channel], mask, [bins], limits).reshape(-1)
            for channel, bins, limits in (
                (0, 16, [0, 180]),
                (1, 4, [0, 256]),
                (2, 4, [0, 256]),
            )
        ]
        descriptor = np.concatenate(histograms).astype(np.float32)
        norm = float(np.linalg.norm(descriptor))
        if norm == 0.0:
            return ()
        return tuple(float(value) for value in descriptor / norm)

    def describe_many(self, detections: Sequence[Detection2DBBox]) -> list[tuple[float, ...]]:
        return [self.describe(detection) for detection in detections]


class DINOAppearanceDescriptor:
    """General-object learned embedding using DimOS's existing DINOv2 model wrapper."""

    def __init__(self) -> None:
        from dimos.models.embedding.dino import DINOModel

        self.model = DINOModel()

    def describe(self, detection: Detection2DBBox) -> tuple[float, ...]:
        return self.describe_many([detection])[0]

    def describe_many(self, detections: Sequence[Detection2DBBox]) -> list[tuple[float, ...]]:
        if not detections:
            return []
        embeddings = self.model.embed(
            *(detection.cropped_image(padding=0) for detection in detections)
        )
        if not isinstance(embeddings, list):
            embeddings = [embeddings]
        return [
            tuple(float(component) for component in embedding.to_numpy().reshape(-1))
            for embedding in embeddings
        ]


class ResNetAppearanceDescriptor:
    """Public ImageNet feature baseline with batched GPU inference."""

    def __init__(self) -> None:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18  # type: ignore[import-untyped]

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(self.device)
        self.preprocess = weights.transforms()

    def describe(self, detection: Detection2DBBox) -> tuple[float, ...]:
        return self.describe_many([detection])[0]

    def describe_many(self, detections: Sequence[Detection2DBBox]) -> list[tuple[float, ...]]:
        if not detections:
            return []
        tensors = [
            self.preprocess(PILImage.fromarray(detection.cropped_image(padding=0).to_rgb().data))
            for detection in detections
        ]
        batch = self._torch.stack(tensors).to(self.device)
        with self._torch.inference_mode():
            features = self.model(batch)
            features = self._torch.nn.functional.normalize(features, dim=-1).cpu().numpy()
        return [tuple(float(component) for component in row) for row in features]


def encode_detection_id(track_id: str, descriptor: Sequence[float]) -> str:
    """Carry an optional float16 descriptor through the standard string ID field."""
    if not descriptor:
        return track_id
    payload = np.asarray(descriptor, dtype=np.float16).tobytes()
    return f"{_PREFIX}{track_id}:{urlsafe_b64encode(payload).decode('ascii')}"


def decode_detection_id(value: str) -> tuple[str, tuple[float, ...]]:
    """Decode IDs produced above; plain third-party IDs remain valid."""
    if not value.startswith(_PREFIX):
        return value, ()
    encoded = value[len(_PREFIX) :]
    track_id, separator, payload = encoded.rpartition(":")
    if not separator:
        return value, ()
    try:
        raw = urlsafe_b64decode(payload.encode("ascii"))
        if len(raw) == 0 or len(raw) % 2 or len(raw) > 8192:
            return value, ()
        descriptor = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    except (ValueError, TypeError):
        return value, ()
    return track_id, tuple(float(component) for component in descriptor)
