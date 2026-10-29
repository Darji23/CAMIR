"""Unit tests for Escalation Tax Accountant (Contribution C1)."""

import pytest
from camir.accounting.tax import EscalationAccountant, EscalationTaxRecord


def test_escalation_tax_record_attributes():
    rec = EscalationTaxRecord(
        prompt_id="p1",
        policy="cascade",
        escalated=True,
        final_tier="large",
        direct_latency_s=0.50,
        direct_cost_usd=0.0002,
        direct_energy_j=0.0,
        actual_latency_s=0.70,   # 0.20s small + 0.50s large
        actual_cost_usd=0.0002,  # small was $0.00
        actual_energy_j=2.5,     # 2.5J small + 0J large
        wasted_latency_s=0.20,
        wasted_cost_usd=0.0,
        wasted_energy_j=2.5,
    )
    assert rec.useful_latency_s == pytest.approx(0.50)
    assert rec.useful_cost_usd == pytest.approx(0.0002)
    assert rec.useful_energy_j == pytest.approx(0.0)
    # Multiplier: 0.70 / 0.50 = 1.40x
    assert abs(rec.latency_tax_multiplier - 1.40) < 1e-4
    assert abs(rec.cost_tax_multiplier - 1.00) < 1e-4


def test_accountant_aggregation():
    acc = EscalationAccountant(policy_name="TestPolicy")

    # Query 1: Not escalated (small accepted)
    acc.record_query(
        prompt_id="q1",
        escalated=False,
        final_tier="small",
        direct_latency_s=0.15,
        direct_cost_usd=0.0,
        direct_energy_j=1.8,
        actual_latency_s=0.15,
        actual_cost_usd=0.0,
        actual_energy_j=1.8,
        wasted_latency_s=0.0,
        wasted_cost_usd=0.0,
        wasted_energy_j=0.0,
    )

    # Query 2: Escalated (small attempt discarded)
    acc.record_query(
        prompt_id="q2",
        escalated=True,
        final_tier="large",
        direct_latency_s=0.50,
        direct_cost_usd=0.0003,
        direct_energy_j=0.0,
        actual_latency_s=0.65,  # 0.15s wasted + 0.50s
        actual_cost_usd=0.0003,
        actual_energy_j=1.8,
        wasted_latency_s=0.15,
        wasted_cost_usd=0.0,
        wasted_energy_j=1.8,
    )

    summary = acc.compute_summary()
    assert summary.total_queries == 2
    assert summary.escalation_count == 1
    assert summary.escalation_rate == 0.50
    assert abs(summary.total_wasted_latency_s - 0.15) < 1e-4
    assert abs(summary.total_wasted_energy_j - 1.8) < 1e-4
    # Multiplier on q1 is 1.0; on q2 is 0.65/0.50 = 1.30 -> average = 1.15x
    assert abs(summary.avg_latency_tax_multiplier - 1.15) < 1e-3
