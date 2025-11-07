"""KServe v2 transformer server.

Wraps preprocess + forward into a FastAPI app exposing the v2 protocol.
KServe routes the request to this container, which calls preprocess()
and forwards to the predictor host given by PREDICTOR_HOST.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import image as img_mod
from . import text as txt_mod

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TASK = os.environ.get("TASK", "image")
PREDICTOR_HOST = os.environ.get("PREDICTOR_HOST", "localhost:8080")
PREDICTOR_URL = f"http://{PREDICTOR_HOST}/v2/models/{{name}}/infer"

app = FastAPI(title="msa-transformer")
_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup() -> None:
    global _client
    _client = httpx.AsyncClient(timeout=30.0)
    if TASK == "text":
        # warm tokenizer
        txt_mod.get_tokenizer()


@app.on_event("shutdown")
async def shutdown() -> None:
    if _client is not None:
        await _client.aclose()


class InferRequest(BaseModel):
    inputs: list[dict]


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "task": TASK}


@app.post("/v2/models/{name}/infer")
async def infer(name: str, req: InferRequest) -> dict:
    t0 = time.perf_counter()
    if TASK == "image":
        blob = req.inputs[0]["data"]
        img = img_mod.decode_image(blob[0] if isinstance(blob, list) else blob)
        arr = img_mod.preprocess(img)
        payload = img_mod.to_v2_payload(arr)
    elif TASK == "text":
        prompt = req.inputs[0]["data"]
        if isinstance(prompt, list):
            prompt = prompt[0]
        payload = txt_mod.to_vllm_chat_payload(prompt, model=name)
    else:
        raise HTTPException(400, f"unknown task {TASK}")

    assert _client is not None
    url = PREDICTOR_URL.format(name=name)
    resp = await _client.post(url, json=payload)
    if resp.status_code >= 400:
        raise HTTPException(resp.status_code, resp.text)

    out = resp.json()
    out.setdefault("metadata", {})["transformer_ms"] = round(
        (time.perf_counter() - t0) * 1000, 2
    )
    return out


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8081"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
