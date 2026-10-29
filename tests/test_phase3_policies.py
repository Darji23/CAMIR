"""Unit tests for Phase 3 active routing policies (P3, P4, P5) and Regime Map."""

import pytest
from camir.guarantees.regime_map import calculate_optimal_regime, generate_regime_grid
from camir.routing.policies import (
    P3SkipRulePolicy,
    P4EarlyAbortPolicy,
    P5VerificationEscalationPolicy,
)
from camir.traces.schema import TraceRow


def make_trace(tier: str, min_logprob: float, latency: float, cost: float) -> TraceRow:
    return TraceRow(
        trace_id=f"t_{tier}",
        prompt_id="test_p",
        dataset="gsm8k",
        tier=tier,
        model_id=f"model_{tier}",
        config_hash="cfg123",
        prompt_text="Math question",
        output_text="42",
        prompt_tokens=30,
        output_tokens=15,
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


def test_p3_skip_rule_policy():
    p3 = P3SkipRulePolicy(rho_threshold=0.40)
    traces = {
        "small": make_trace("small", min_logprob=-0.9, latency=0.20, cost=0.0),
        "large": make_trace("large", min_logprob=-0.1, latency=0.60, cost=0.001),
    }

    # Highly complex math and code prompt -> low P(success) -> skip small
    hard_prompt = {
        "prompt": "def solve_differential_equation(x, y): calculate integral = 14 * 28 + 99 - 142 / 85 return formula"
    }
    tier, row, meta = p3.route(hard_prompt, traces)
    assert tier == "large"
    assert meta["skipped_small"] is True
    assert meta["wasted_latency_s"] == 0.0  # Zero wasted small work!

    # Simple factual prompt -> high P(success) -> try small
    simple_prompt = {"prompt": "What color is grass?"}
    tier_s, row_s, meta_s = p3.route(simple_prompt, traces)
    assert meta_s["skipped_small"] is False


def test_p4_early_abort_policy():
    p4 = P4EarlyAbortPolicy(tau_abort=-1.0, n0_tokens=10)
    traces = {
        "small": make_trace("small", min_logprob=-1.8, latency=0.30, cost=0.0),
        "large": make_trace("large", min_logprob=-0.1, latency=0.60, cost=0.001),
    }
    tier, row, meta = p4.route({"prompt": "Hard query"}, traces)
    assert tier == "large"
    assert meta["aborted_early"] is True
    # Wasted latency must be less than full 0.30s
    assert meta["wasted_latency_s"] < 0.30
    assert meta["saved_decode_latency_s"] > 0.0


def test_p5_verification_escalation_policy():
    p5 = P5VerificationEscalationPolicy(tau_gate=-0.65, lambda_verifier=-2.0)
    traces = {
        "small": make_trace("small", min_logprob=-0.85, latency=0.20, cost=0.0),
        "large": make_trace("large", min_logprob=-0.1, latency=0.60, cost=0.001),
    }
    traces["small"].mean_logprob = -1.2  # v_score (-1.2) >= lambda (-2.0)

    tier, row, meta = p5.route({"prompt": "Query"}, traces)
    # Verifier saved the small answer!
    assert tier == "small"
    assert meta["saved_by_verifier"] is True
    assert meta["escalated"] is False


def test_regime_map_boundary_calculations():
    # Extreme escalation rate (e=0.90) with high cost ratio -> P5 or P3 dominates P2
    pt_high_e = calculate_optimal_regime(cost_ratio_rho=30.0, escalation_rate_e=0.85)
    assert pt_high_e.optimal_policy in ("P5-Verification", "P3-SkipRule")
    assert pt_high_e.net_savings_pct > 0.0

    # Very low escalation rate (e=0.05) -> P2 Cascade is highly efficient
    pt_low_e = calculate_optimal_regime(cost_ratio_rho=20.0, escalation_rate_e=0.05)
    assert pt_low_e.optimal_policy in ("P2-Cascade", "P5-Verification")
