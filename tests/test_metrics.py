from src import metrics as M


def test_render_returns_bytes_and_content_type():
    body, ctype = M.render()
    assert isinstance(body, (bytes, bytearray))
    assert "text/plain" in ctype
    # registry has all expected metric families
    text = body.decode()
    for name in (
        "msa_request_latency_seconds",
        "msa_requests_total",
        "msa_fallback_total",
        "msa_queue_depth",
        "msa_cold_starts_total",
    ):
        assert name in text, f"missing metric {name}"


def test_counter_increments_visible_in_render():
    M.REQUESTS_TOTAL.labels(model="t", status="ok").inc(2)
    body, _ = M.render()
    assert b'msa_requests_total{model="t",status="ok"} 2' in body
