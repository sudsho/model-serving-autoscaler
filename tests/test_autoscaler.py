"""Unit tests for the autoscaler control logic (pure, no cluster)."""
from pathlib import Path

from src import autoscaler as A

CFG = Path(__file__).resolve().parents[1] / "configs" / "per_model.yaml"


def _policy(**kw) -> A.ScalingPolicy:
    base = dict(
        name="m",
        metric="rps",
        target_per_replica=50.0,
        min_replicas=0,
        max_replicas=8,
        scale_down_delay_s=60.0,
    )
    base.update(kw)
    return A.ScalingPolicy(**base)


def test_raw_desired_is_ceil_of_load_over_target():
    p = _policy()
    assert p.raw_desired(0) == 0
    assert p.raw_desired(1) == 1      # ceil(1/50)
    assert p.raw_desired(50) == 1
    assert p.raw_desired(51) == 2
    assert p.raw_desired(200) == 4


def test_raw_desired_clamps_to_bounds():
    p = _policy(min_replicas=1, max_replicas=4)
    assert p.raw_desired(0) == 1          # never below min
    assert p.raw_desired(10_000) == 4     # never above max


def test_scale_up_is_immediate():
    p = _policy()
    auto = A.Autoscaler(policy=p)
    assert auto.replicas == 0
    # a burst of load scales up on the very next tick, no delay
    assert auto.step(t=0, observed_total=200) == 4


def test_scale_down_waits_for_stabilization_window():
    p = _policy(scale_down_delay_s=60.0)
    auto = A.Autoscaler(policy=p)
    auto.step(t=0, observed_total=200)      # -> 4 replicas
    # load drops to idle at t=10; the window is measured from there
    assert auto.step(t=10, observed_total=0) == 4
    assert auto.step(t=60, observed_total=0) == 4   # 50s < 60s window
    # once the full 60s window elapses we drop straight to the floor
    assert auto.step(t=70, observed_total=0) == 0


def test_scale_down_timer_resets_when_load_returns():
    p = _policy(scale_down_delay_s=60.0)
    auto = A.Autoscaler(policy=p)
    auto.step(t=0, observed_total=200)      # 4 replicas
    auto.step(t=30, observed_total=0)       # pending scale-down (timer at 30)
    # load comes back before the window closes -> stay up, timer cancelled
    assert auto.step(t=40, observed_total=200) == 4
    # now idle again from t=99; must wait a fresh full window
    assert auto.step(t=99, observed_total=0) == 4
    assert auto.step(t=158, observed_total=0) == 4   # 59s < 60s
    assert auto.step(t=159, observed_total=0) == 0   # window closed


def test_never_scales_below_min_on_idle():
    p = _policy(min_replicas=2, max_replicas=6, scale_down_delay_s=30.0)
    auto = A.Autoscaler(policy=p)
    auto.step(t=0, observed_total=300)      # -> 6
    auto.step(t=0, observed_total=0)
    assert auto.step(t=60, observed_total=0) == 2   # floor at min, not 0


def test_from_config_parses_all_models():
    policies = A.load_policies(CFG)
    assert set(policies) == {"sklearn-churn", "pytorch-resnet50", "llm-llama3-8b"}
    sk = policies["sklearn-churn"]
    assert sk.metric == "rps"
    assert sk.target_per_replica == 50
    assert sk.min_replicas == 0 and sk.max_replicas == 8
    assert sk.scale_down_delay_s == 60.0  # parsed from "60s"
    llm = policies["llm-llama3-8b"]
    assert llm.metric == "queue_depth"
    assert llm.scale_down_delay_s == 300.0


def test_parse_seconds_variants():
    assert A._parse_seconds("90s", 0) == 90.0
    assert A._parse_seconds("2m", 0) == 120.0
    assert A._parse_seconds("500ms", 0) == 0.5
    assert A._parse_seconds(45, 0) == 45.0
    assert A._parse_seconds(None, 7) == 7.0
    assert A._parse_seconds("garbage", 7) == 7.0


def test_apply_to_cluster_is_dry_run_without_client():
    p = _policy()
    msg = A.apply_to_cluster(p, 3, client=None)
    assert msg.startswith("[dry-run]")
    assert "3 replicas" in msg


def test_full_episode_rises_then_returns_to_floor():
    """End-to-end: rising then falling load scales up and back down."""
    p = _policy(min_replicas=0, max_replicas=8, scale_down_delay_s=60.0)
    auto = A.Autoscaler(policy=p)
    loads = [0, 50, 150, 400, 400, 200, 50, 0, 0, 0, 0]
    tl = [auto.step(t=i * 30, observed_total=x) for i, x in enumerate(loads)]
    assert max(tl) == 8           # saturated at max under peak load
    assert tl[0] <= 1             # started idle
    assert tl[-1] == 0            # drained back to floor when idle
