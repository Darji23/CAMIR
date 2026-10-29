"""Unit tests for offline ReplayEngine and baseline routing policies."""

import pytest
from camir.replay.baselines import (
    AlwaysLargePolicy,
    AlwaysMediumPolicy,
    AlwaysSmallPolicy,
    ComplexityRouterPolicy,
    FeatureRouterPolicy,
    LogprobCascadePolicy,
    OraclePolicy,
    RandomMixPolicy,
)
from camir.replay.engine import ReplayEngine
from camir.traces.schema import TimingBreakdown, TraceRow


def make_mock_trace(
    pid: str,
    tier: str,
    output_text: str,
    cost: float,
    latency: float,
    min_logprob: float,
) -> TraceRow:
    return TraceRow(
        trace_id=f"trace_{pid}_{tier}",
        prompt_id=pid,
        dataset="arc_easy",
        tier=tier,
        model_id=f"model_{tier}",
        config_hash="hash123",
        prompt_text="Question: ... Answer:",
        output_text=output_text,
        prompt_tokens=20,
        output_tokens=10,
        min_logprob=min_logprob,
        cost_usd=cost,
        t_prefill_s=latency * 0.2,
        t_ttft_s=latency * 0.2,
        t_decode_s=latency * 0.8,
        t_total_s=latency,
        tpot_s=0.01,
        energy_j=1.0 if tier != "large" else 0.0,
        energy_method="estimated" if tier != "large" else "none",
        hardware_profile="apple_silicon" if tier != "large" else "cloud",
        hardware_info="Apple Silicon",
    )


def test_baseline_routing_decisions():
    prompt_item = {"prompt_id": "p1", "dataset": "arc_easy", "prompt": "Question: 1+1? Answer:", "reference_answer": "B"}
    traces = {
        "small": make_mock_trace("p1", "small", "Wrong answer A", cost=0.0, latency=0.1, min_logprob=-0.9),
        "medium": make_mock_trace("p1", "medium", "Right answer B", cost=0.0, latency=0.3, min_logprob=-0.3),
        "large": make_mock_trace("p1", "large", "Right answer B", cost=0.0005, latency=0.5, min_logprob=-0.1),
    }

    # 1. Always Small
    tier_s, row_s, _ = AlwaysSmallPolicy().route(prompt_item, traces)
    assert tier_s == "small"

    # 2. Always Large
    tier_l, row_l, _ = AlwaysLargePolicy().route(prompt_item, traces)
    assert tier_l == "large"

    # 3. Oracle chooses medium (cheapest correct)
    tier_o, row_o, meta_o = OraclePolicy().route(prompt_item, traces)
    assert tier_o == "medium"
    assert meta_o["oracle_matched"] is True

    # 4. Logprob Cascade escalates because min_logprob (-0.9) < tau (-0.5)
    cascade = LogprobCascadePolicy(threshold_min_lp=-0.50)
    tier_c, row_c, meta_c = cascade.route(prompt_item, traces)
    assert tier_c == "large"
    assert meta_c["escalated"] is True
    assert meta_c["wasted_latency_s"] == 0.1  # small tier latency wasted


def test_replay_engine_tax_accounting():
    """Verify that escalation tax correctly accumulates small and large latency/cost."""
    class DummyStore:
        pass

    engine = ReplayEngine(DummyStore())
    prompts = [
        {"prompt_id": "p1", "dataset": "arc_easy", "prompt": "Q1", "reference_answer": "A"},
        {"prompt_id": "p2", "dataset": "arc_easy", "prompt": "Q2", "reference_answer": "B"},
    ]
    traces = {
        "p1": {
            # High confidence small -> accepted without escalation
            "small": make_mock_trace("p1", "small", "A", cost=0.0, latency=0.10, min_logprob=-0.1),
            "large": make_mock_trace("p1", "large", "A", cost=0.001, latency=0.50, min_logprob=-0.05),
        },
        "p2": {
            # Low confidence small -> rejected and escalated to large
            "small": make_mock_trace("p2", "small", "C", cost=0.0, latency=0.15, min_logprob=-1.2),
            "large": make_mock_trace("p2", "large", "B", cost=0.001, latency=0.60, min_logprob=-0.05),
        },
    }

    cascade = LogprobCascadePolicy(threshold_min_lp=-0.50)
    result = engine.evaluate_policy(cascade, prompts, traces)

    assert result.total_queries == 2
    assert result.accuracy == 1.0  # Both p1 (small) and p2 (escalated to large) get correct answer!
    # p1 cost = $0.00; p2 cost = $0.001 -> total = $0.001
    assert abs(result.total_cost_usd - 0.001) < 1e-6
    # p1 latency = 0.10; p2 latency = 0.15 (small wasted) + 0.60 (large) = 0.75 -> mean = (0.10 + 0.75) / 2 = 0.425
    assert abs(result.mean_latency_s - 0.425) < 1e-4
    # Wasted latency must be 0.15
    assert abs(result.wasted_latency_s - 0.15) < 1e-4
    # Escalation tax ratio > 1.0
    assert result.escalation_tax_ratio > 1.0
