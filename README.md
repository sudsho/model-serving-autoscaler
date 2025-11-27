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

`pytest` covers the router, transformers, cold-start retry, canary helpers
and config schema. None of them touch a real cluster.

```bash
make test
```

## CI

`ci/test.yml.example` runs ruff + pytest + a yaml-lint pass on the
manifests, then builds the transformer image. Move it under
`.github/workflows/test.yml` once your repo has the workflow scope.

## License

MIT.
