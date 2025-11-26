import asyncio

import httpx
import pytest

from src import cold_start


def test_is_cold_response_503():
    req = httpx.Request("GET", "http://x")
    r = httpx.Response(503, request=req)
    assert cold_start.is_cold_response(r)


def test_is_cold_response_504():
    req = httpx.Request("GET", "http://x")
    assert cold_start.is_cold_response(httpx.Response(504, request=req))


def test_is_cold_response_ok_not_cold():
    req = httpx.Request("GET", "http://x")
    assert not cold_start.is_cold_response(httpx.Response(200, request=req))


@pytest.mark.asyncio
async def test_with_retry_recovers_after_one_503():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="cold")
        return httpx.Response(200, text="warm")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await cold_start.with_retry(client, "GET", "http://x", retries=1, base_delay=0.01)
    assert resp.status_code == 200
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_with_retry_gives_up_after_budget():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="cold")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await cold_start.with_retry(client, "GET", "http://x", retries=2, base_delay=0.001)
    assert resp.status_code == 503
