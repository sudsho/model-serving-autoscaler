#!/usr/bin/env bash
# Roll a canary forward in stages. Eval must pass between each step.
#
# Stages: 10% -> 25% -> 50% -> 100%
# usage: canary/promote.sh sklearn-churn data/eval/churn_eval.jsonl
set -euo pipefail

SVC="${1:?service name}"
EVAL_DS="${2:?eval dataset path}"
NS="${NS:-msa}"

stages=(10 25 50 100)

for pct in "${stages[@]}"; do
  stable=$((100 - pct))
  echo "==> promoting $SVC: stable=$stable canary=$pct"
  bash scripts/traffic_split.sh "$SVC" "$stable" "$pct"

  echo "==> waiting 90s for traffic to stabilize"
  sleep 90

  base="http://${SVC}-stable.${NS}.example.com"
  cand="http://${SVC}-canary.${NS}.example.com"
  if ! python -m canary.eval --baseline "$base" --canary "$cand" --dataset "$EVAL_DS"; then
    echo "!! eval failed at $pct%, rolling back"
    bash scripts/traffic_split.sh "$SVC" 100 0
    exit 1
  fi
done

echo "promotion complete"
