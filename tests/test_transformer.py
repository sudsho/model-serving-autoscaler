"""Tests for the image + text transformer modules.

These run without any cluster, so they're safe to exercise in CI.
"""
import base64
import io

import numpy as np
import pytest
from PIL import Image

from src.transformer import image as imgmod


def _encoded_image(size: int = 32) -> str:
    img = Image.new("RGB", (size, size), color=(123, 0, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_decode_image_base64():
    enc = _encoded_image()
    img = imgmod.decode_image(enc)
    assert img.size == (32, 32)
    assert img.mode == "RGB"


def test_decode_image_bytes():
    enc = _encoded_image()
    raw = base64.b64decode(enc)
    img = imgmod.decode_image(raw)
    assert img.size == (32, 32)


def test_decode_image_bad_base64_raises():
    with pytest.raises(ValueError):
        imgmod.decode_image("not-base64!!")


def test_preprocess_shape_and_dtype():
    img = Image.new("RGB", (10, 10))
    arr = imgmod.preprocess(img, size=224)
    assert arr.shape == (3, 224, 224)
    assert arr.dtype == np.float32


def test_preprocess_normalization_within_expected_range():
    img = Image.new("RGB", (224, 224), color=(127, 127, 127))
    arr = imgmod.preprocess(img)
    # not clipped, but should be small in absolute terms after normalization
    assert np.abs(arr).max() < 3.0


def test_to_v2_payload_shape_round_trip():
    arr = np.zeros((3, 224, 224), dtype=np.float32)
    payload = imgmod.to_v2_payload(arr, name="x")
    assert payload["inputs"][0]["name"] == "x"
    assert payload["inputs"][0]["shape"] == [3, 224, 224]
    assert payload["inputs"][0]["datatype"] == "FP32"
    assert len(payload["inputs"][0]["data"]) == 3 * 224 * 224
