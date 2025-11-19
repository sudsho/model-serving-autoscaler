"""Cold start handling.

Knative scale-to-zero is great for cost but the first request after a
pod has been GC'd has a big tail. We mitigate two ways:

1. activator-aware retries: when KServe's activator buffers the request
   we sometimes get a 503 with a retry hint; the router treats this as
   transient and retries once with backoff.
2. opportunistic warm-up: a background task pings each ksvc on a slow
   cadence so it stays in the "active" state during business hours.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Iterable

import httpx

logger = logging.getLogger(__name__)


# headers KServe / activator typically use to flag a buffered request
_RETRY_HINT_HEADERS = ("k-revision", "x-envoy-overloaded")


def is_cold_response(resp: httpx.Response) -> bool:
    if resp.status_code in (503, 504):
        return True
    return any(h in resp.headers for h in _RETRY_HINT_HEADERS) and resp.status_code == 503


async def with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    retries: int = 1,
    base_delay: float = 0.3,
    **kwargs,
) -> httpx.Response:
    last: httpx.Response | None = None
    for i in range(retries + 1):
        last = await client.request(method, url, **kwargs)
        if not is_cold_response(last):
            return last
        if i == retries:
            return last
        delay = base_delay * (2**i) + random.uniform(0, 0.1)
        logger.info("cold start hit on %s, retrying in %.2fs", url, delay)
        await asyncio.sleep(delay)
    assert last is not None
    return last


async def warmer_loop(
    client: httpx.AsyncClient,
    services: Iterable[str],
    *,
    interval_s: float = 90.0,
    gateway: str | None = None,
) -> None:
    gw = gateway or os.environ.get("KSERVE_GATEWAY", "http://localhost")
    while True:
        for svc in services:
            try:
                await client.get(
                    f"{gw}/v2/models/{svc}",
                    headers={"Host": f"{svc}.msa.example.com"},
                    timeout=5.0,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("warm ping %s failed: %s", svc, e)
        await asyncio.sleep(interval_s)
