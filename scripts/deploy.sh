#!/usr/bin/env bash
# Apply all KServe + Knative manifests in order.
# Assumes kubectl context already points at the right cluster and
# kserve + knative are installed.
set -euo pipefail

NS="${NS:-msa}"

echo "==> ensuring namespace $NS"
kubectl apply -f kserve/namespace.yaml

echo "==> ServingRuntimes"
kubectl apply -f kserve/serving-runtimes.yaml

echo "==> InferenceServices"
kubectl apply -n "$NS" -f kserve/sklearn-churn.yaml
kubectl apply -n "$NS" -f kserve/pytorch-resnet50.yaml
kubectl apply -n "$NS" -f kserve/llm-llama3.yaml

echo "==> Knative routes"
kubectl apply -n "$NS" -f knative/sklearn-canary.yaml
kubectl apply -n "$NS" -f knative/ab-routing.yaml

echo "==> waiting for InferenceServices to be ready"
for svc in sklearn-churn pytorch-resnet50 llm-llama3-8b; do
  kubectl wait -n "$NS" --for=condition=Ready inferenceservice/"$svc" --timeout=10m \
    || echo "WARN: $svc not ready yet, check kubectl describe"
done

echo "done."
