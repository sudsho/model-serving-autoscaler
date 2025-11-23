"""Router tests: use FastAPI's TestClient with httpx mocked at the
transport layer so we never touch the cluster.
"""
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from src import router as router_mod


@pytest.fixture
def cfg(tmp_path: Path):
    cfg_path = tmp_path / "per_model.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "sklearn-churn": {"type": "sklearn"},
                    "pytorch-resnet50": {"type": "pytorch"},
                },
                "routing": {"fallback_chain": {"pytorch-resnet50": "sklearn-churn"}},
            }
        ),
        encoding="utf-8",
    )
    return cfg_path


@pytest.fixture
def client(monkeypatch, cfg):
    monkeypatch.setattr(router_mod, "CONFIG_PATH", cfg)
    # rebuild module config
    router_mod._cfg = router_mod.load_config()  # type: ignore[attr-defined]

    transport_calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        transport_calls.append(request)
        if "pytorch" in request.url.path:
            # simulate predictor down
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(
            200,
            json={"outputs": [{"name": "y", "data": [0]}]},
        )

    fake = httpx.MockTransport(handler)
    router_mod._client = httpx.AsyncClient(transport=fake)  # type: ignore[attr-defined]
    return TestClient(router_mod.app), transport_calls


def test_healthz(client):
    c, _ = client
    r = c.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "sklearn-churn" in body["models"]


def test_predict_unknown_model_404(client):
    c, _ = client
    r = c.post("/v1/predict/nope", json={"inputs": [{"name": "x", "data": [0]}]})
    assert r.status_code == 404


def test_predict_metadata_present(client):
    c, _ = client
    r = c.post(
        "/v1/predict/sklearn-churn", json={"inputs": [{"name": "x", "data": [0]}]}
    )
    assert r.status_code == 200
    out = r.json()
    assert out["metadata"]["served_by"] == "sklearn-churn"
    assert out["metadata"]["model_type"] == "sklearn"
    assert "router_ms" in out["metadata"]


def test_predict_falls_back_when_primary_down(client):
    c, calls = client
    r = c.post(
        "/v1/predict/pytorch-resnet50", json={"inputs": [{"name": "x", "data": [0]}]}
    )
    assert r.status_code == 200
    served = r.json()["metadata"]["served_by"]
    assert served == "sklearn-churn"
    # primary attempted, then fallback
    assert any("pytorch" in str(call.url) for call in calls)
    assert any("sklearn" in str(call.url) for call in calls)
