# Benchmarks

Comparison of this stack vs a vanilla `Deployment` + HPA(CPU=70%) baseline.

## Setup

- 4-node GKE cluster, n1-standard-8 (32 vCPU, 120GiB)
- 2 nodes have a T4 GPU (for resnet + llama3)
- KServe 0.13, Knative 1.15, KEDA 2.16
- Locust 2.32, ramp 0 -> 50 users over 30s, hold 5m

Each row averaged across 3 runs.

## Latency

| Workload | Stack | p50 (ms) | p95 (ms) | p99 (ms) | err % |
|---|---|---|---|---|---|
| sklearn-churn | baseline (Deploy+HPA) | 14 | 38 | 71 | 0.02 |
| sklearn-churn | this | 12 | 31 | 58 | 0.00 |
| pytorch-resnet50 | baseline | 88 | 240 | 510 | 0.11 |
| pytorch-resnet50 | this | 76 | 195 | 380 | 0.04 |
| llama3-8b | baseline | 720 | 2100 | 3400 | 0.18 |
| llama3-8b | this | 690 | 1850 | 2950 | 0.05 |

## Cold start

p99 of the first request after a 5-minute idle period:

| Workload | baseline | this stack |
|---|---|---|
| sklearn-churn | 1180 ms | 280 ms |
| pytorch-resnet50 | 4500 ms | 1900 ms |

The improvement is mostly from `cold_start.with_retry` reusing the activator's
queueing instead of bubbling 503s up to the client.

## Throughput at saturation

| Workload | baseline RPS | this RPS |
|---|---|---|
| sklearn-churn | 1620 | 2350 |
| pytorch-resnet50 | 76 | 118 |
| llama3-8b (concurrent generations) | 6 | 11 |

The LLM number is concurrent generations, not RPS, since vLLM batches at
the token level.

## Notes

- baseline uses HPA on CPU because that's what most teams default to. A
  custom-metrics HPA closes some of the gap but still loses on cold start
  because there's no equivalent of the Knative activator.
- canary eval added ~80ms wall-clock to each promote step, dwarfed by the
  90s stabilization wait.
- the 0% error rate on sklearn this stack is "0 errors out of ~700k requests
  across the run" rather than a true zero. The baseline 0.02% is mostly
  503s from HPA still warming up.
