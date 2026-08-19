"""Autoscaler control logic (pure, offline-simulatable).

On a real cluster the scaling decisions are made by Knative's KPA (for the
sklearn and pytorch InferenceServices) and by KEDA (for the vLLM LLM, bridging
`vllm_running_requests` from Prometheus). Both boil down to the same core
control law, which this module implements in plain Python so it can be
unit-tested and simulated without a cluster:

    desired = ceil(observed_total_metric / target_per_replica)

clamped to ``[min_replicas, max_replicas]``, with a stabilization window so we
scale *up* immediately but only scale *down* after the load has stayed low for
``scale_down_delay`` seconds. That "up fast, down slow" asymmetry is exactly
what Knative's KPA does and is what keeps bursty inference traffic from
flapping replicas.

The per-model targets and bounds are read from the same
``configs/per_model.yaml`` that the router and the deploy script use, so the
simulation exercises the real production knobs.

Nothing here imports the kubernetes client, KEDA, or boto3. Applying the
computed replica count to a live cluster is the cluster's job (Knative/KEDA);
see ``README.md`` for how the metric is wired in production. The
``apply_to_cluster`` helper below is intentionally guarded and is a no-op
unless a real client is injected.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# metric name -> human unit, used only for pretty printing
_METRIC_UNITS = {
    "rps": "req/s",
    "concurrency": "in-flight",
    "queue_depth": "queued",
}

# some configs express the scale-down delay as e.g. "60s"; parse loosely.
def _parse_seconds(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    try:
        if s.endswith("ms"):
            return float(s[:-2]) / 1000.0
        if s.endswith("s"):
            return float(s[:-1])
        if s.endswith("m"):
            return float(s[:-1]) * 60.0
        return float(s)
    except ValueError:
        return default


@dataclass(frozen=True)
class ScalingPolicy:
    """Immutable per-model scaling parameters (mirrors the yaml autoscaler block)."""

    name: str
    metric: str
    target_per_replica: float
    min_replicas: int
    max_replicas: int
    scale_down_delay_s: float = 60.0

    @classmethod
    def from_config(cls, name: str, spec: dict[str, Any]) -> "ScalingPolicy":
        auto = spec.get("autoscaler", {}) or {}
        metric = auto.get("metric", "concurrency")
        target = float(auto.get("target", spec.get("target_concurrency", 1)) or 1)
        if target <= 0:
            target = 1.0
        return cls(
            name=name,
            metric=metric,
            target_per_replica=target,
            min_replicas=int(spec.get("min_replicas", 0)),
            max_replicas=int(spec.get("max_replicas", 1)),
            scale_down_delay_s=_parse_seconds(auto.get("scale_down_delay"), 60.0),
        )

    def raw_desired(self, observed_total: float) -> int:
        """The unstabilized target: proportional to load, clamped to bounds."""
        if observed_total <= 0:
            want = self.min_replicas
        else:
            want = math.ceil(observed_total / self.target_per_replica)
        return max(self.min_replicas, min(self.max_replicas, want))

    def unit(self) -> str:
        return _METRIC_UNITS.get(self.metric, self.metric)


@dataclass
class Autoscaler:
    """Stateful KPA-style controller for a single model.

    Call :meth:`step` once per observation tick with the wall-clock time and the
    total observed metric across all replicas. It returns the replica count the
    controller wants *now*, applying the up-fast / down-slow stabilization.
    """

    policy: ScalingPolicy
    replicas: int = field(default=0)
    # timestamp we first saw a lower-than-current desired; None while at/above.
    _cooldown_since: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # boot at min (or 1 if the model must never scale to zero)
        if self.replicas <= 0:
            self.replicas = max(self.policy.min_replicas, 0)
        self.replicas = max(self.policy.min_replicas, min(self.policy.max_replicas, self.replicas))

    def step(self, t: float, observed_total: float) -> int:
        raw = self.policy.raw_desired(observed_total)

        if raw > self.replicas:
            # scale up immediately, cancel any pending scale-down
            self.replicas = raw
            self._cooldown_since = None
        elif raw < self.replicas:
            # candidate scale-down: only commit after the stabilization window
            # has fully elapsed with the target staying below current. Once it
            # does, drop straight to the current desired (like the KPA acting on
            # the trailing-window max), rather than trickling down one at a time.
            if self._cooldown_since is None:
                self._cooldown_since = t
            elif (t - self._cooldown_since) >= self.policy.scale_down_delay_s:
                self.replicas = raw
                self._cooldown_since = None
        else:
            # exactly on target; hold and clear the timer
            self._cooldown_since = None

        return self.replicas


def load_policies(config_path: str | os.PathLike[str]) -> dict[str, ScalingPolicy]:
    """Build one :class:`ScalingPolicy` per model from ``per_model.yaml``."""
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    models = data.get("models", {}) or {}
    return {name: ScalingPolicy.from_config(name, spec) for name, spec in models.items()}


def apply_to_cluster(policy: ScalingPolicy, replicas: int, *, client: Any = None) -> str:
    """Guarded hook for applying a decision to a real cluster.

    In production the replica count is not pushed by this process at all: Knative
    and KEDA own the InferenceService scale. This helper exists only so an
    operator experiment could patch a Deployment directly. It refuses to do
    anything unless a real kubernetes client is explicitly injected, so importing
    or simulating this module never touches a cluster or the network.
    """
    if client is None:
        return f"[dry-run] would set {policy.name} -> {replicas} replicas"
    # real patch path, only reachable when a caller passes a live client.
    client.patch_namespaced_deployment_scale(  # pragma: no cover - needs cluster
        name=policy.name,
        namespace="msa",
        body={"spec": {"replicas": replicas}},
    )
    return f"patched {policy.name} -> {replicas} replicas"
