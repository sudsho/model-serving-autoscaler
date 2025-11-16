"""Canary evaluation: compare baseline vs canary on a fixed test set
before promoting traffic.

Usage:
    python -m canary.eval \
        --baseline http://sklearn-churn-stable.msa.example.com \
        --canary   http://sklearn-churn-canary.msa.example.com \
        --dataset  data/eval/churn_eval.jsonl

Returns exit code 0 if canary passes, non-zero otherwise. The promote
script reads the exit code and decides whether to shift traffic.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median, quantiles

import httpx


@dataclass
class Outcome:
    n: int
    errors: int
    p50_ms: float
    p95_ms: float
    agreement: float  # how often canary's top1 == baseline's top1


def hit(client: httpx.Client, base_url: str, payload: dict) -> tuple[float, dict]:
    t = time.perf_counter()
    r = client.post(f"{base_url}/v2/models/sklearn-churn/infer", json=payload, timeout=10.0)
    elapsed = (time.perf_counter() - t) * 1000.0
    if r.status_code >= 400:
        return elapsed, {"error": r.status_code}
    return elapsed, r.json()


def run_one(client: httpx.Client, base_url: str, ds: list[dict]) -> Outcome:
    lat: list[float] = []
    out: list[dict] = []
    errors = 0
    for row in ds:
        ms, body = hit(client, base_url, row["payload"])
        if "error" in body:
            errors += 1
        lat.append(ms)
        out.append(body)
    qs = quantiles(lat, n=20) if len(lat) >= 20 else [median(lat), max(lat)]
    return Outcome(
        n=len(lat),
        errors=errors,
        p50_ms=median(lat),
        p95_ms=qs[18] if len(qs) > 18 else qs[-1],
        agreement=0.0,
    )


def agreement(a: list[dict], b: list[dict]) -> float:
    same = 0
    n = min(len(a), len(b))
    for i in range(n):
        ta = (a[i].get("outputs") or [{}])[0].get("data", [None])[0]
        tb = (b[i].get("outputs") or [{}])[0].get("data", [None])[0]
        if ta == tb:
            same += 1
    return same / n if n else 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--canary", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--p95-budget-ms", type=float, default=250.0)
    p.add_argument("--err-budget", type=float, default=0.01)
    p.add_argument("--min-agreement", type=float, default=0.97)
    args = p.parse_args()

    rows = [json.loads(l) for l in Path(args.dataset).read_text().splitlines() if l.strip()]
    with httpx.Client() as c:
        base = run_one(c, args.baseline, rows)
        cand = run_one(c, args.canary, rows)

    # rerun once to compute agreement on outputs
    with httpx.Client() as c:
        a_out = [hit(c, args.baseline, r["payload"])[1] for r in rows]
        b_out = [hit(c, args.canary, r["payload"])[1] for r in rows]
    agree = agreement(a_out, b_out)
    cand.agreement = agree

    print(json.dumps({"baseline": vars(base), "canary": vars(cand)}, indent=2))

    fails = []
    if cand.p95_ms > args.p95_budget_ms:
        fails.append(f"p95 {cand.p95_ms:.1f}ms > budget {args.p95_budget_ms}ms")
    if cand.errors / max(1, cand.n) > args.err_budget:
        fails.append(f"error rate {cand.errors}/{cand.n} > budget")
    if agree < args.min_agreement:
        fails.append(f"agreement {agree:.3f} < {args.min_agreement}")
    if fails:
        print("CANARY FAILED:", "; ".join(fails), file=sys.stderr)
        return 1
    print("canary ok, safe to promote.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
