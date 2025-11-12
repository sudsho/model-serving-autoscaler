#!/usr/bin/env bash
# Patch a Knative service traffic block.
#
# usage:
#   scripts/traffic_split.sh sklearn-churn 50 50
#       -> stable=50, canary=50
#   scripts/traffic_split.sh pytorch-resnet50 100 0
#       -> promote stable, drop canary
set -euo pipefail

SVC="${1:?service name}"
STABLE_PCT="${2:?stable percent}"
CANARY_PCT="${3:?canary percent}"
NS="${NS:-msa}"

if (( STABLE_PCT + CANARY_PCT != 100 )); then
  echo "stable+canary must equal 100, got $STABLE_PCT + $CANARY_PCT" >&2
  exit 1
fi

patch=$(cat <<EOF
[
  {"op": "replace", "path": "/spec/traffic/0/percent", "value": $STABLE_PCT},
  {"op": "replace", "path": "/spec/traffic/1/percent", "value": $CANARY_PCT}
]
EOF
)

kubectl patch ksvc "$SVC" -n "$NS" --type=json -p "$patch"
kubectl get ksvc "$SVC" -n "$NS" -o jsonpath='{.status.traffic}{"\n"}'
