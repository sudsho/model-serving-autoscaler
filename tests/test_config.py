"""Static config validation: parse the per-model yaml and assert
the required fields that the deploy script depends on.
"""
from pathlib import Path

import yaml

CFG = Path(__file__).resolve().parents[1] / "configs" / "per_model.yaml"


def test_config_loads():
    data = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    assert data is not None
    assert "models" in data


def test_each_model_has_required_fields():
    data = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    for name, spec in data["models"].items():
        for k in ("type", "storage_uri", "min_replicas", "max_replicas"):
            assert k in spec, f"{name} missing {k}"
        assert spec["min_replicas"] <= spec["max_replicas"]


def test_autoscaler_metric_known():
    data = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    valid = {"rps", "concurrency", "queue_depth"}
    for name, spec in data["models"].items():
        m = spec["autoscaler"]["metric"]
        assert m in valid, f"{name} has unknown metric {m}"


def test_canary_thresholds_sane():
    data = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    for name, spec in data["models"].items():
        c = spec.get("canary", {})
        if c.get("enabled"):
            t = c["promote_threshold"]
            assert t["p95_latency_ms"] > 0
            assert 0 <= t["error_rate"] <= 0.1
