# model-serving-autoscaler

KServe-based multi-model serving with autoscaling on Kubernetes.

## What

Production model serving stack for sklearn, PyTorch and vLLM-served LLMs. Uses
KServe `InferenceService` per model, Knative for scale-to-zero and traffic
splitting, custom transformers for preprocessing.

Targets:

- autoscale by inference RPS, queue depth and p95 latency (not just CPU)
- canary deploys with eval-before-promote
- A/B routing via Knative weighted traffic splits
- cold start handling so first-request latency stays sane
- per-model concurrency and resource caps

Status: WIP. Bench numbers in `benchmarks/results.md`.

## Stack

- KServe 0.13
- Kubernetes 1.31
- Knative Serving
- FastAPI (router and transformer sidecars)
- Prometheus + Grafana 11
- Locust for load gen
