#!/usr/bin/env bash
# Package a trained model artifact into the layout each runtime expects
# and push to the configured object store URI.
#
# usage:
#   scripts/model_pkg.sh sklearn ./out/churn-v3 gs://msa-models/sklearn/churn-v3
#   scripts/model_pkg.sh pytorch ./out/resnet50 gs://msa-models/pytorch/resnet50-v2
set -euo pipefail

KIND="${1:?kind: sklearn|pytorch|hf}"
SRC="${2:?source dir}"
DST="${3:?destination uri (gs:// or s3:// or hf://)}"

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

case "$KIND" in
  sklearn)
    # KServe sklearnserver expects model.joblib at root
    cp "$SRC/model.joblib" "$stage/"
    ;;
  pytorch)
    # torchserve expects a .mar archive
    if [ ! -f "$SRC/model.mar" ]; then
      echo "missing $SRC/model.mar, run torch-model-archiver first" >&2
      exit 1
    fi
    mkdir -p "$stage/model-store" "$stage/config"
    cp "$SRC/model.mar" "$stage/model-store/"
    cp "$SRC/config.properties" "$stage/config/" 2>/dev/null || \
      printf 'inference_address=http://0.0.0.0:8085\n' > "$stage/config/config.properties"
    ;;
  hf)
    cp -R "$SRC/." "$stage/"
    ;;
  *)
    echo "unknown kind $KIND" >&2; exit 2;;
esac

echo "uploading $stage -> $DST"
case "$DST" in
  gs://*)
    gsutil -m rsync -r "$stage" "$DST"
    ;;
  s3://*)
    aws s3 sync "$stage" "$DST"
    ;;
  hf://*)
    echo "hf:// uploads done out-of-band via huggingface-cli login + push" >&2
    ;;
  *)
    echo "unsupported scheme $DST" >&2; exit 3;;
esac

echo "ok."
