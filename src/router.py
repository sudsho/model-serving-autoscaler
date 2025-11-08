"""Top-level FastAPI router.

Sits in front of KServe so callers don't have to know per-model URLs.
Reads `configs/per_model.yaml`, decides which InferenceService to hit,
falls back to a stub when the predictor is cold or down.

Routes
------
GET  /healthz
GET  /models                  -> list models known to the router
POST /v1/predict/{model}      -> forward to the matching KServe URL
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CONFIG_PATH = Path(os.environ.get("ROUTER_CONFIG", "configs/per_model.yaml"))
KSERVE_GW = os.environ.get(
    "KSERVE_GATEWAY", "http://istio-ingressgateway.istio-system.svc.cluster.local"
)
DEFAULT_TIMEOUT = float(os.environ.get("ROUTER_TIMEOUT_S", "30"))

app = FastAPI(title="msa-router")
_client: httpx.AsyncClient | None = None
_cfg: dict[str, Any] = {}


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        logger.warning("router config %s missing, using empty config", CONFIG_PATH)
        return {"models": {}, "routing": {}}
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


@app.on_event("startup")
async def startup() -> None:
    global _client, _cfg
    _client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
    _cfg = load_config()
    logger.info("router up; %d model(s) registered", len(_cfg.get("models", {})))


@app.on_event("shutdown")
async def shutdown() -> None:
    if _client is not None:
        await _client.aclose()


class PredictRequest(BaseModel):
    inputs: list[dict]
    parameters: dict | None = None


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"status": "ok", "models": list(_cfg.get("models", {}).keys())}


@app.get("/models")
async def list_models() -> dict[str, Any]:
    return {"models": _cfg.get("models", {})}


def _kserve_url(model: str) -> str:
    # KServe Serverless: HOST header routes to the right ksvc.
    # We hit the gateway and set Host below.
    return f"{KSERVE_GW}/v2/models/{model}/infer"


@app.post("/v1/predict/{model}")
async def predict(model: str, req: PredictRequest) -> dict[str, Any]:
    if model not in _cfg.get("models", {}):
        raise HTTPException(404, f"unknown model {model}")

    spec = _cfg["models"][model]
    namespace = "msa"
    host = f"{model}.{namespace}.example.com"

    t0 = time.perf_counter()
    assert _client is not None
    try:
        resp = await _client.post(
            _kserve_url(model),
            json=req.dict(),
            headers={"Host": host},
        )
    except httpx.HTTPError as e:
        # try fallback chain if configured
        fallback = _cfg.get("routing", {}).get("fallback_chain", {}).get(model)
        if fallback is not None:
            logger.warning("primary %s failed (%s); falling back to %s", model, e, fallback)
            return await predict(fallback, req)  # type: ignore[arg-type]
        raise HTTPException(503, f"predictor unreachable: {e}") from e

    if resp.status_code >= 400:
        raise HTTPException(resp.status_code, resp.text)

    out = resp.json()
    out.setdefault("metadata", {})["router_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    out["metadata"]["served_by"] = model
    out["metadata"]["model_type"] = spec.get("type", "unknown")
    return out


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
