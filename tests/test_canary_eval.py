"""Smoke test for canary evaluator helpers (no network calls)."""
from canary import eval as ce


def test_agreement_perfect_match():
    a = [{"outputs": [{"name": "y", "data": [1]}]} for _ in range(5)]
    b = [{"outputs": [{"name": "y", "data": [1]}]} for _ in range(5)]
    assert ce.agreement(a, b) == 1.0


def test_agreement_zero_when_all_differ():
    a = [{"outputs": [{"name": "y", "data": [1]}]} for _ in range(4)]
    b = [{"outputs": [{"name": "y", "data": [0]}]} for _ in range(4)]
    assert ce.agreement(a, b) == 0.0


def test_agreement_handles_empty_lists():
    assert ce.agreement([], []) == 0.0


def test_agreement_partial():
    a = [{"outputs": [{"name": "y", "data": [i]}]} for i in [1, 0, 1, 1]]
    b = [{"outputs": [{"name": "y", "data": [i]}]} for i in [1, 1, 1, 0]]
    # matches at indices 0, 2 -> 2/4
    assert ce.agreement(a, b) == 0.5
