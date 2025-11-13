"""Locust load generator targeting the router.

Run with:
    locust -f src/load_test.py --headless -u 50 -r 10 -t 5m \
        --host http://router.msa.example.com

Tasks weighted to mimic a realistic mix: lots of sklearn churn requests
(cheap, high QPS), fewer image classification, even fewer LLM prompts.
"""
from __future__ import annotations

import base64
import io
import random

from locust import HttpUser, between, task
from PIL import Image


def _fake_image(size: int = 224) -> str:
    img = Image.new("RGB", (size, size), color=(random.randint(0, 255),) * 3)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


_PROMPTS = [
    "Summarize the second law of thermodynamics in two sentences.",
    "Write a haiku about distributed systems.",
    "List five benefits of vector databases.",
    "Explain LSM trees in one paragraph.",
    "What is autoscaling in Knative?",
]

_CHURN_FEATURES = [
    {"tenure": 12, "monthly_charges": 79.5, "contract": "month-to-month"},
    {"tenure": 24, "monthly_charges": 64.2, "contract": "one-year"},
    {"tenure": 60, "monthly_charges": 105.0, "contract": "two-year"},
]


class ServingUser(HttpUser):
    wait_time = between(0.05, 0.5)

    @task(70)
    def hit_sklearn(self) -> None:
        feats = random.choice(_CHURN_FEATURES)
        self.client.post(
            "/v1/predict/sklearn-churn",
            json={"inputs": [{"name": "input__0", "data": [list(feats.values())]}]},
            name="POST /predict/sklearn-churn",
        )

    @task(20)
    def hit_pytorch(self) -> None:
        self.client.post(
            "/v1/predict/pytorch-resnet50",
            json={"inputs": [{"name": "input__0", "data": _fake_image()}]},
            name="POST /predict/pytorch-resnet50",
        )

    @task(10)
    def hit_llm(self) -> None:
        self.client.post(
            "/v1/predict/llm-llama3-8b",
            json={"inputs": [{"name": "prompt", "data": random.choice(_PROMPTS)}]},
            name="POST /predict/llm-llama3-8b",
        )
