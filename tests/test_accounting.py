"""Unit tests for timing, cost, and energy accounting."""

import pytest
from camir.accounting.cost import calculate_token_cost
from camir.accounting.energy import EnergyTracker
from camir.accounting.timing import HighResTimer, compute_timing_breakdown


def test_hand_computed_token_cost_accounting():
    """Verify cost = tokens * price against a hand-computed tariff.
    
    Tariff:
    - Hosted Large Tier (qwen/qwen-2.5-72b-instruct):
      $0.35 per 1,000,000 prompt tokens ($0.00000035 / token)
      $0.40 per 1,000,000 completion tokens ($0.00000040 / token)
      
    Hand calculation:
      Prompt tokens: 1,420
      Completion tokens: 380
      
      Prompt cost:     1,420 * 0.00000035 = 0.00049700 USD
      Completion cost:   380 * 0.00000040 = 0.00015200 USD
      --------------------------------------------------
      Total Expected:                     0.00064900 USD
    """
    prompt_tokens = 1420
    completion_tokens = 380
    hand_computed_expected = 0.00064900

    actual_cost = calculate_token_cost("qwen/qwen-2.5-72b-instruct", prompt_tokens, completion_tokens)
    assert abs(actual_cost - hand_computed_expected) < 1e-8, (
        f"Cost calculation mismatch! Expected ${hand_computed_expected:.8f}, got ${actual_cost:.8f}"
    )


def test_cost_calculation():
    # Local models must always cost $0.00
    assert calculate_token_cost("mlx-community/Qwen2.5-1.5B-Instruct-4bit", 1000, 200) == 0.0
    assert calculate_token_cost("mlx-community/Qwen2.5-7B-Instruct-4bit", 1000, 200) == 0.0
    assert calculate_token_cost("local-small", 500, 100) == 0.0

    # Hosted Qwen-72B pricing test ($0.35/1M input, $0.40/1M output)
    # 1,000,000 in + 1,000,000 out = $0.75
    cost = calculate_token_cost("qwen/qwen-2.5-72b-instruct", 1_000_000, 1_000_000)
    assert abs(cost - 0.75) < 1e-5

    # Small token batch: 1000 in, 500 out
    # 0.001 * 0.35 + 0.0005 * 0.40 = 0.00035 + 0.00020 = 0.00055
    cost_small = calculate_token_cost("qwen/qwen-2.5-72b-instruct", 1000, 500)
    assert abs(cost_small - 0.00055) < 1e-6


def test_timing_breakdown():
    timing = compute_timing_breakdown(
        t_load_s=1.5,
        t_prefill_s=0.04,
        t_ttft_s=0.045,
        t_decode_s=0.20,
        output_tokens=21,
        cache_disabled=True,
        thermal_state="nominal",
    )
    assert timing.t_load_s == 1.5
    assert timing.t_prefill_s == 0.04
    assert timing.t_ttft_s == 0.045
    assert timing.t_decode_s == 0.20
    assert abs(timing.t_total_s - 0.24) < 1e-5
    # TPOT: 0.20 / (21 - 1) = 0.010 s/token
    assert abs(timing.tpot_s - 0.010) < 1e-5
    assert timing.cache_disabled is True
    assert timing.thermal_state == "nominal"


def test_energy_tracker():
    with EnergyTracker(use_powermetrics=False) as tracker:
        # Simulate small work
        _ = sum(i for i in range(100_000))

    assert tracker.energy_j is not None
    assert tracker.energy_j >= 0.0
    assert tracker.energy_method == "estimated"
    assert "estimated, 12.5W" in tracker.label
