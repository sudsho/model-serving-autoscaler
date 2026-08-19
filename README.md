# model-serving-autoscaler

KServe model serving with autoscaling, multi-model support, deployed on
Kubernetes.

## Problem

Vanilla `Deployment` + HPA on CPU works for stateless web apps. It does not
work well for ML serving:

- a sklearn pod and an LLM pod don't want the same scaling rule
- HPA on CPU is a poor proxy for "this thing is overloaded with inference"
- cold starts hurt p99 latency on bursty traffic
- canary rollouts need eval-before-promote, not just a percent slider
- you want to fall back when the GPU pod evicts, not 503

This repo wires KServe + Knative to handle all of that for three model
shapes (sklearn, PyTorch image classifier, vLLM-served LLM) behind one
FastAPI router.

## Quick start (runs offline)

You do not need a cluster, a GPU, or a network to see the two headline pieces
work. The offline smoke boots the FastAPI router in-process against a synthetic
predictor and drives the autoscaler control loop over a synthetic load curve.

```bash
pip install -r requirements.txt   # torch is NOT needed for the smoke
make smoke                        # or: python scripts/smoke.py
```

Real output (trimmed; idle stretches collapsed by the script):

```
====================================================================
PART A  serving API in-process (FastAPI router + synthetic predictor)
====================================================================
  /healthz ok, 3 models registered: sklearn-churn, pytorch-resnet50, llm-llama3-8b
  10/10 synthetic requests -> 200, router p50=0.42ms
  unknown model -> 404 as expected
  /metrics scrape ok (prometheus exposition served)

====================================================================
PART B  autoscaler control logic (pure simulation, no cluster)
====================================================================

  [sklearn-churn]  metric=rps target=50/replica  bounds=[0,8]  scale_down_delay=60s
    t=   0s  load   66.7 req/s     |......                                  | replicas= 2 ##
    t=  90s  load  266.7 req/s     |..........................              | replicas= 6 ######
    t= 150s  load  400.0 req/s     |........................................| replicas= 8 ########
    t= 210s  load  400.0 req/s     |........................................| replicas= 8 ########
    t= 300s  load  200.0 req/s     |....................                    | replicas= 4 ####
    t= 390s  load    0.0 req/s     |                                        | replicas= 0
    t= 540s  load    0.0 req/s     |                                        | replicas= 0
      [ok ] scaled up under load
      [ok ] respected max_replicas
      [ok ] respected min_replicas
      [ok ] scaled down after peak
      [ok ] settled at floor when idle
      -> replicas start=2 peak=8 end=0

  [pytorch-resnet50]  metric=concurrency target=2/replica  bounds=[1,12]  scale_down_delay=90s
    ... replicas start=2 peak=12 end=1  (all checks ok)

  [llm-llama3-8b]  metric=queue_depth target=16/replica  bounds=[1,4]  scale_down_delay=300s
    ... replicas start=1 peak=4 end=1  (all checks ok)

  cluster apply hook is guarded: '[dry-run] would set sklearn-churn -> 3 replicas'

====================================================================
SMOKE OK  serving API answered synthetic requests; autoscaler scaled
          replicas up under load and back to floor when idle.
====================================================================
```

What the smoke exercises, and what it stubs:

- **Part A** runs the real `src/router.py` FastAPI app via `TestClient`. A tiny
  in-process "model" (a fixed logistic on the churn feature vector) is mocked at
  the httpx transport layer in place of the KServe predictors, so real v2
  inference requests flow through the router: dispatch, metadata, `/metrics`,
  and the 404 path. No cluster, no network.
- **Part B** runs the real scaling control law from `src/autoscaler.py` against
  the per-model targets in `configs/per_model.yaml`. This is the same
  "desired = ceil(load / target_per_replica), up fast / down after a
  stabilization window" law that Knative's KPA and KEDA apply on the cluster,
  extracted so it can be simulated and unit-tested. The load curve rises to
  saturation and falls back to idle; the smoke asserts the controller scales up
  under load, honors `min`/`max`, and drains back to the floor (scale-to-zero
  for sklearn) when idle.

### What still needs a real cluster / GPU

The smoke proves the control *logic* and the serving path. It does not stand up
the production system. For the real thing you need:

- Kubernetes 1.31 + KServe 0.13 + Knative Serving to actually apply the replica
  counts and route traffic (the `kserve/` and `knative/` manifests).
- KEDA + Prometheus to bridge `vllm_running_requests` into the LLM autoscaler.
- A GPU node pool for the resnet50 and llama3 predictors.
- The `benchmarks/results.md` numbers were measured on a 4-node GKE cluster and
  are not reproduced by the offline smoke.

`src/autoscaler.apply_to_cluster` is the only function that would touch a
cluster; it is a guarded dry-run unless a live kubernetes client is injected, so
nothing here imports the kubernetes client or opens a socket.

## Architecture

```
                    +-------------------+
                    |   client / SDK    |
                    +---------+---------+
                              |
                              v
                  +-----------+-----------+
                  |   FastAPI router      |
                  |   (src/router.py)     |
                  |   - per-model dispatch|
                  |   - fallback chain    |
                  |   - cold-start retry  |
                  |   - prom /metrics     |
                  +-----+-----+-----+-----+
                        |     |     |
            +-----------+     |     +-----------+
            v                 v                 v
+-------------------+ +-------------------+ +-------------------+
| transformer (img) | | transformer (txt) | | transformer (txt) |
|  preprocess       | |  tokenize         | |  tokenize         |
+---------+---------+ +---------+---------+ +---------+---------+
          v                     v                     v
+-------------------+ +-------------------+ +-------------------+
|  sklearnserver    | |  torchserve       | |  vllm runtime     |
|  churn-v3         | |  resnet50-v2      | |  llama3-8b        |
| min=0 max=8 rps=50| | min=1 max=12 c=2  | | min=1 max=4       |
+-------------------+ +-------------------+ +-------------------+

   ^ Knative scale-to-zero            ^ KEDA on vllm_running_requests
```

## Stack

- KServe 0.13 with sklearnserver, torchserve, vllm runtimes
- Kubernetes 1.31, Knative Serving for autoscale + traffic split
- KEDA bridges Prometheus metrics into the autoscaler for the LLM
- FastAPI 0.116 for router and transformer sidecars
- Prometheus + Grafana 11 for observability
- Locust 2.32 for load gen

## Layout

```
.
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|-- Dockerfile                 (transformer sidecar image)
|-- Makefile
|-- LICENSE
|-- configs/
|   `-- per_model.yaml         (single source of truth for the router + scripts)
|-- kserve/
|   |-- namespace.yaml
|   |-- serving-runtimes.yaml
|   |-- sklearn-churn.yaml     (InferenceService)
|   |-- pytorch-resnet50.yaml  (InferenceService + transformer)
|   |-- llm-llama3.yaml        (vLLM-backed InferenceService)
|   `-- autoscaler-tuning.yaml (Knative configmap + KEDA scaler for vLLM)
|-- knative/
|   |-- sklearn-canary.yaml    (90/10 stable/canary split)
|   `-- ab-routing.yaml        (tag-based A/B for QA bypass)
|-- src/
|   |-- router.py
|   |-- autoscaler.py          (KPA-style scaling control law, offline-simulatable)
|   |-- metrics.py
|   |-- cold_start.py
|   |-- load_test.py           (locust)
|   `-- transformer/
|       |-- image.py
|       |-- text.py
|       `-- server.py
|-- canary/
|   |-- eval.py                (latency + agreement gate)
|   `-- promote.sh             (10 -> 25 -> 50 -> 100 staged roll)
|-- monitoring/
|   |-- prometheus-scrape.yaml
|   |-- grafana-dashboard.json
|   `-- alerts.yaml
|-- scripts/
|   |-- smoke.py               (offline end-to-end smoke: router + autoscaler sim)
|   |-- deploy.sh
|   |-- model_pkg.sh
|   |-- traffic_split.sh
|   |-- train_churn.py
|   `-- train_resnet.py
|-- benchmarks/
|   `-- results.md
|-- tests/
|-- ci/
|   `-- test.yml.example
```

## Quickstart

```bash
# 1. cluster prerequisites
#    kubernetes 1.31+, kserve 0.13, knative serving, prometheus operator
make install

# 2. package and upload the artefacts
python scripts/train_churn.py --out out/churn-v3
bash scripts/model_pkg.sh sklearn out/churn-v3 gs://msa-models/sklearn/churn-v3

# 3. apply the cluster manifests
bash scripts/deploy.sh

# 4. run the router locally for smoke testing
python -m src.router

# 5. push some load
locust -f src/load_test.py --headless -u 50 -r 10 -t 5m \
       --host http://router.msa.example.com
```

## Autoscaling: which signal for which model?

| Model | Metric | Target / pod | Why |
|---|---|---|---|
| sklearn-churn | rps | 50 | feature vectors are tiny, CPU bound; rps tracks load 1:1 |
| pytorch-resnet50 | concurrency | 2 | GPU saturates at low concurrency; rps overcounts batched calls |
| llm-llama3-8b | vllm_running_requests | 16 | token-level queueing; KEDA bridges the metric to the autoscaler |

## Canary

`canary/eval.py` runs the same eval set against baseline and candidate, then
checks three gates:

1. p95 latency under the per-model SLO (`canary.promote_threshold.p95_latency_ms`)
2. error rate under `error_rate`
3. output agreement >= 0.97 (so you don't promote a model that classifies
   half the dataset differently for no reason)

`canary/promote.sh` rolls 10 -> 25 -> 50 -> 100 with eval between each step
and rolls back on the first failure.

## Cold start handling

Two layers:

- `src/cold_start.with_retry` looks at the response (503 + activator headers)
  and retries once with backoff. The retry usually lands on a freshly-warm
  pod since the activator queued the original.
- `src/cold_start.warmer_loop` is an optional background task that pings
  every InferenceService at a slow cadence so the activator keeps them
  "active" during business hours. Off by default; flip with `WARMER=1`.

## Benchmarks

See `benchmarks/results.md`. Headline numbers comparing this stack to a
plain `Deployment` + HPA(CPU=70%) baseline at 50 concurrent users:

| Workload | p50 (ms) | p95 (ms) | Cold start p99 (ms) |
|---|---|---|---|
| sklearn-churn (Deployment+HPA) | 14 | 38 | 1180 |
| sklearn-churn (this stack) | 12 | 31 | 280 |
| pytorch-resnet50 (Deployment+HPA) | 88 | 240 | 4500 |
| pytorch-resnet50 (this stack) | 76 | 195 | 1900 |
| llama3-8b (Deployment+HPA) | 720 | 2100 | n/a (always-on) |
| llama3-8b (this stack) | 690 | 1850 | n/a (min=1) |

## Tests

`pytest` covers the router, transformers, cold-start retry, canary helpers,
the autoscaler control law, and the config schema. None of them touch a real
cluster.

```bash
make test        # 35 tests, all offline
```

Real output:

```
$ python -m pytest tests -q
...................................                                       [100%]
35 passed in 1.18s
```

## CI

`ci/test.yml.example` runs ruff + pytest + a yaml-lint pass on the
manifests, then builds the transformer image. Move it under
`.github/workflows/test.yml` once your repo has the workflow scope.

## License

MIT.
