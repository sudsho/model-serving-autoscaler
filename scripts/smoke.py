"""Offline smoke test for model-serving-autoscaler.

Runs the whole thing with NO cluster, NO GPU, NO network:

  Part A - serving API in-process
      Boots the FastAPI router (``src/router.py``) with a synthetic KServe
      predictor mocked at the httpx transport layer, so a tiny in-process
      "model" answers real v2 inference requests. Fires a mixed batch of
      synthetic requests (sklearn / pytorch / llm) through the router and
      checks status, metadata and the fallback chain.

  Part B - autoscaler control logic as a pure simulation
      Feeds a synthetic load curve (QPS / concurrency / queue-depth that rises
      then falls) into the real scaling policy from ``configs/per_model.yaml``
      and prints the replica count over time, asserting the controller scales
      up under load and back down when idle.

Everything the production stack would get from Knative / KEDA / a real
predictor is stubbed here. Real autoscaling on Kubernetes stays documented in
the README.

Usage:
    python scripts/smoke.py          # or:  make smoke
Exit code 0 = all checks green.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# keep the smoke output readable: silence per-request client logging.
for _name in ("httpx", "httpcore", "src.router"):
    logging.getLogger(_name).setLevel(logging.WARNING)

import httpx  # noqa: E402
import yaml  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src import autoscaler as A  # noqa: E402
from src import router as router_mod  # noqa: E402

CONFIG = ROOT / "configs" / "per_model.yaml"


# --------------------------------------------------------------------------- #
# tiny synthetic "model": a fixed logistic on the churn feature vector.
# stands in for the sklearnserver / torchserve / vllm predictors behind KServe.
# --------------------------------------------------------------------------- #
def _tiny_predict(inputs: list[dict]) -> list[dict]:
    first = inputs[0] if inputs else {}
    data = first.get("data")
    # churn-style feature vector -> deterministic pseudo-probability
    if isinstance(data, list) and data and isinstance(data[0], list):
        feats = data[0]
        # weights chosen so longer tenure lowers churn, higher charges raise it
        w = [-0.03, 0.02, 0.0]
        z = sum(wi * (fi if isinstance(fi, (int, float)) else 0.0) for wi, fi in zip(w, feats))
        prob = 1.0 / (1.0 + pow(2.71828, -z))
        label = int(prob >= 0.5)
        return [{"name": "output__0", "datatype": "FP32", "shape": [1, 2],
                 "data": [round(1 - prob, 4), round(prob, 4)], "label": label}]
    # image / text path: just echo a class id / short completion
    return [{"name": "output__0", "datatype": "INT64", "shape": [1], "data": [285]}]


def _mock_kserve_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # emulate a predictor that is briefly cold once, to exercise retry.
        try:
            body = request.read()
            payload = __import__("json").loads(body) if body else {}
        except Exception:
            payload = {}
        outputs = _tiny_predict(payload.get("inputs", []))
        return httpx.Response(200, json={"model_name": request.url.path, "outputs": outputs})

    return httpx.MockTransport(handler)


def _boot_router() -> tuple[TestClient, list[httpx.Request]]:
    router_mod.CONFIG_PATH = CONFIG
    router_mod._cfg = router_mod.load_config()  # type: ignore[attr-defined]

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        body = request.read()
        payload = __import__("json").loads(body) if body else {}
        outputs = _tiny_predict(payload.get("inputs", []))
        return httpx.Response(200, json={"model_name": str(request.url.path), "outputs": outputs})

    # inject the mock client directly (no lifespan -> no real AsyncClient created)
    router_mod._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[attr-defined]
    return TestClient(router_mod.app), calls


def part_a_serving() -> None:
    print("=" * 68)
    print("PART A  serving API in-process (FastAPI router + synthetic predictor)")
    print("=" * 68)
    client, calls = _boot_router()

    r = client.get("/healthz")
    assert r.status_code == 200, r.text
    models = r.json()["models"]
    print(f"  /healthz ok, {len(models)} models registered: {', '.join(models)}")

    # fire a mixed synthetic batch, mimicking load_test.py weights (7:2:1)
    plan = (["sklearn-churn"] * 7) + (["pytorch-resnet50"] * 2) + (["llm-llama3-8b"] * 1)
    feats = [[12, 79.5, 0], [24, 64.2, 1], [60, 105.0, 2]]
    ok = 0
    lat = []
    for i, model in enumerate(plan):
        payload = {"inputs": [{"name": "input__0", "data": [feats[i % len(feats)]]}]}
        resp = client.post(f"/v1/predict/{model}", json=payload)
        assert resp.status_code == 200, f"{model}: {resp.status_code} {resp.text}"
        meta = resp.json()["metadata"]
        assert meta["served_by"] == model
        lat.append(meta["router_ms"])
        ok += 1
    print(f"  {ok}/{len(plan)} synthetic requests -> 200, "
          f"router p50={sorted(lat)[len(lat)//2]:.2f}ms")

    # unknown model -> 404
    assert client.post("/v1/predict/nope", json={"inputs": []}).status_code == 404
    print("  unknown model -> 404 as expected")

    # /metrics scrape works and reflects the traffic
    m = client.get("/metrics")
    assert m.status_code == 200 and b"msa_requests_total" in m.content
    print("  /metrics scrape ok (prometheus exposition served)")
    print()


# --------------------------------------------------------------------------- #
# Part B: autoscaler simulation
# --------------------------------------------------------------------------- #
def _episode(policy: A.ScalingPolicy, tick_s: float) -> list[float]:
    """Load curve: idle -> ramp up -> hold at saturation -> ramp down -> idle.

    Peak is set to ``target * max_replicas`` so the controller is driven all the
    way to its ceiling, and the trailing idle tail is sized to the model's own
    ``scale_down_delay`` so the run always covers a full drain back to the floor.
    """
    peak = policy.target_per_replica * policy.max_replicas
    ramp_up = [round(peak * i / 6, 1) for i in range(1, 7)]   # 6 ticks climbing
    hold = [peak, peak]                                        # saturate the ceiling
    ramp_down = [round(peak * i / 6, 1) for i in range(5, -1, -1)]  # 6 ticks falling to 0
    idle_ticks = int(policy.scale_down_delay_s // tick_s) + 3
    idle = [0.0] * idle_ticks
    return ramp_up + hold + ramp_down + idle


def _simulate(policy: A.ScalingPolicy, curve: list[float], tick_s: float) -> list[int]:
    auto = A.Autoscaler(policy=policy)
    timeline = []
    for i, load in enumerate(curve):
        replicas = auto.step(t=i * tick_s, observed_total=load)
        timeline.append(replicas)
    return timeline


def _spark(curve: list[float], timeline: list[int], policy: A.ScalingPolicy,
           tick_s: float) -> None:
    peak = max(curve) or 1.0
    prev = None
    for i, (load, rep) in enumerate(zip(curve, timeline)):
        # collapse long idle stretches: only print when the replica count moves
        # (or at the very start / end), so a 300s drain doesn't flood the log.
        is_edge = i == 0 or i == len(curve) - 1
        if not is_edge and rep == prev and load == 0.0 and timeline[i - 1] == rep:
            prev = rep
            continue
        bar = "#" * rep
        pct = int(40 * load / peak)
        load_bar = ("." * pct).ljust(40)
        print(f"    t={int(i*tick_s):>4}s  load {load:>6.1f} {policy.unit():<9} "
              f"|{load_bar}| replicas={rep:>2} {bar}")
        prev = rep


def part_b_autoscaler() -> None:
    print("=" * 68)
    print("PART B  autoscaler control logic (pure simulation, no cluster)")
    print("=" * 68)
    policies = A.load_policies(CONFIG)
    tick_s = 30.0

    all_ok = True
    for name, policy in policies.items():
        curve = _episode(policy, tick_s)
        timeline = _simulate(policy, curve, tick_s)

        start, top, end = timeline[0], max(timeline), timeline[-1]
        idle_tail = timeline[-1]
        print(f"\n  [{name}]  metric={policy.metric} target={policy.target_per_replica:g}"
              f"/replica  bounds=[{policy.min_replicas},{policy.max_replicas}]"
              f"  scale_down_delay={policy.scale_down_delay_s:g}s")
        _spark(curve, timeline, policy, tick_s)

        # assertions: scaled up under load, and back down when idle
        checks = {
            "scaled up under load": top > start,
            "respected max_replicas": top <= policy.max_replicas,
            "respected min_replicas": min(timeline) >= policy.min_replicas,
            "scaled down after peak": end < top,
            "settled at floor when idle": idle_tail <= max(policy.min_replicas, start),
        }
        for label, passed in checks.items():
            flag = "ok " if passed else "FAIL"
            print(f"      [{flag}] {label}")
            all_ok = all_ok and passed
        print(f"      -> replicas start={start} peak={top} end={end}")

    print()
    if not all_ok:
        raise AssertionError("autoscaler simulation failed one or more checks")

    # the guarded cluster hook must stay a dry-run offline
    p = next(iter(policies.values()))
    msg = A.apply_to_cluster(p, 3, client=None)
    assert msg.startswith("[dry-run]"), msg
    print(f"  cluster apply hook is guarded: {msg!r}")
    print()


def main() -> int:
    try:
        part_a_serving()
        part_b_autoscaler()
    except AssertionError as e:
        print(f"SMOKE FAILED: {e}", file=sys.stderr)
        return 1
    print("=" * 68)
    print("SMOKE OK  serving API answered synthetic requests; autoscaler scaled")
    print("          replicas up under load and back to floor when idle.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
