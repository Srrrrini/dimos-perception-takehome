# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import numpy as np
import pytest

from dimos.msgs.sensor_msgs.Image import Image
from dimos.perception.detection.appearance import (
    HSVHistogramDescriptor,
    decode_detection_id,
    encode_detection_id,
)
from dimos.perception.detection.type.detection2d.seg import Detection2DSeg


def test_descriptor_transport_preserves_track_id_and_vector() -> None:
    encoded = encode_detection_id("track:7", (0.25, -0.5, 1.0))

    track_id, descriptor = decode_detection_id(encoded)

    assert track_id == "track:7"
    assert descriptor == pytest.approx((0.25, -0.5, 1.0), abs=5e-4)


def test_plain_detection_id_remains_backward_compatible() -> None:
    assert decode_detection_id("42") == ("42", ())


def test_hsv_descriptor_uses_instance_mask_and_is_normalized() -> None:
    pixels = np.full((20, 20, 3), (255, 0, 0), dtype=np.uint8)
    pixels[5:15, 5:15] = (0, 0, 255)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    detection = Detection2DSeg(
        bbox=(0.0, 0.0, 20.0, 20.0),
        track_id=1,
        class_id=0,
        confidence=0.9,
        name="object",
        ts=1.0,
        image=Image.from_opencv(pixels, ts=1.0),
        mask=mask,
    )

    descriptor = HSVHistogramDescriptor().describe(detection)

    assert len(descriptor) == 24
    assert np.linalg.norm(descriptor) == pytest.approx(1.0)
    assert descriptor[0] > descriptor[10]  # red hue, not blue background
