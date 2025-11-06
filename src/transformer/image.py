"""Image preprocessing transformer.

Runs as a sidecar in front of the predictor. Accepts raw image bytes
(jpeg/png) or base64, normalizes to a tensor in the shape the model
expects, and forwards to the predictor host.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# imagenet stats; switch per model if you ever bring something else
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def decode_image(blob: str | bytes) -> Image.Image:
    if isinstance(blob, str):
        # base64 path
        try:
            blob = base64.b64decode(blob)
        except Exception as e:
            raise ValueError(f"could not decode base64: {e}") from e
    return Image.open(io.BytesIO(blob)).convert("RGB")


def preprocess(img: Image.Image, size: int = 224) -> np.ndarray:
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(2, 0, 1)  # CHW
    return arr.astype(np.float32)


def to_v2_payload(arr: np.ndarray, name: str = "input__0") -> dict[str, Any]:
    return {
        "inputs": [
            {
                "name": name,
                "shape": list(arr.shape),
                "datatype": "FP32",
                "data": arr.flatten().tolist(),
            }
        ]
    }
