"""Unit tests for trace-driven discrete-event queuing simulator (Experiment E12)."""

import pytest
from camir.replay.baselines import AlwaysLargePolicy, AlwaysSmallPolicy, OraclePolicy
from camir.simulation.load_simulator import DiscreteEventSimulator, SimulationMetrics
from camir.traces.schema import TimingBreakdown, TraceRow


@pytest.fixture
def sample_traces():
    timing_s = TimingBreakdown(
        t_load_s=0.0,
        t_prefill_s=0.05,
        t_ttft_s=0.05,
        t_decode_s=0.10,
        t_total_s=0.15,
        tpot_s=0.01,
        cache_disabled=True,
        thermal_state="nominal",
    )
    timing_l = TimingBreakdown(
        t_load_s=0.0,
        t_prefill_s=0.20,
        t_ttft_s=0.20,
        t_decode_s=0.10,
        t_total_s=0.30,
        tpot_s=0.01,
        cache_disabled=True,
        thermal_state="nominal",
    )
    row_s = TraceRow(
        trace_id="t-s-1",
        prompt_id="prompt-1",
        dataset="arc_easy",
        tier="small",
        model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        config_hash="h1",
        prompt_text="Q1",
        output_text="A",
        prompt_tokens=10,
        output_tokens=10,
        cost_usd=0.0,
        t_prefill_s=0.05,
        t_ttft_s=0.05,
        t_decode_s=0.10,
        t_total_s=0.15,
        tpot_s=0.01,
        energy_j=0.4,
        energy_method="estimated",
        hardware_profile="apple_silicon",
        hardware_info="M4",
    )
    row_l = TraceRow(
        trace_id="t-l-1",
        prompt_id="prompt-1",
        dataset="arc_easy",
        tier="large",
        model_id="accounts/fireworks/models/qwen3p8-max",
        config_hash="h2",
        prompt_text="Q1",
        output_text="A",
        prompt_tokens=10,
        output_tokens=10,
        cost_usd=0.0001,
        t_prefill_s=0.20,
        t_ttft_s=0.20,
        t_decode_s=0.10,
        t_total_s=0.30,
        tpot_s=0.01,
        energy_j=0.0,
        energy_method="none",
        hardware_profile="cloud",
        hardware_info="Cloud",
    )
    return {"prompt-1": {"small": row_s, "large": row_l}}


@pytest.fixture
def sample_prompts():
    return [
        {"prompt_id": "prompt-1", "dataset": "arc_easy", "prompt": "Q1", "reference_answer": "A"}
    ]


def test_simulation_low_load(sample_prompts, sample_traces):
    sim = DiscreteEventSimulator(local_concurrency=1, hosted_concurrency=5, seed=42)
    pol = AlwaysSmallPolicy()
    metrics = sim.run_simulation(
        policy=pol,
        prompt_items=sample_prompts,
        traces_by_prompt=sample_traces,
        arrival_rate_qps=0.5,
        duration_s=10.0,
    )
    assert isinstance(metrics, SimulationMetrics)
    assert metrics.total_queries > 0
    assert metrics.p50_latency_s > 0.0
    assert metrics.p95_latency_s >= metrics.p50_latency_s


def test_simulation_queue_scaling(sample_prompts, sample_traces):
    sim = DiscreteEventSimulator(local_concurrency=1, hosted_concurrency=5, seed=42)
    pol = AlwaysSmallPolicy()
    low_load = sim.run_simulation(pol, sample_prompts, sample_traces, arrival_rate_qps=0.5, duration_s=20.0)
    high_load = sim.run_simulation(pol, sample_prompts, sample_traces, arrival_rate_qps=4.0, duration_s=20.0)

    assert high_load.total_queries > low_load.total_queries
    # Under high arrival rate on single local server, mean wait time rises
    assert high_load.mean_wait_s >= low_load.mean_wait_s
