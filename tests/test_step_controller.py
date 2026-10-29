"""Unit tests for StepController and multi-step agentic routing (Contribution C4 / E9)."""

import pytest
from camir.routing.step_controller import AgentStep, StepController, TrajectoryResult


def test_step_controller_initial_decision():
    ctrl = StepController(max_small_context_tokens=1000)
    # Short context routes to small
    assert ctrl.decide_step_tier(step_index=1, context_tokens=250) == "small"
    # Long context skips small and routes directly to large
    assert ctrl.decide_step_tier(step_index=4, context_tokens=1500) == "large"


def test_step_controller_success_retention():
    ctrl = StepController(tau_step_confidence=-0.50)
    step = ctrl.evaluate_step_execution(
        step_index=1,
        prompt_text="Call search tool",
        context_tokens=200,
        initial_tier="small",
        small_result={"min_logprob": -0.25, "cost_usd": 0.0, "latency_s": 0.15, "text": "search('cats')"},
        large_result={"cost_usd": 0.001, "latency_s": 0.60, "text": "search('cats')"},
    )
    assert step.tier_chosen == "small"
    assert step.escalated is False
    assert step.wasted_cost_usd == 0.0


def test_step_controller_dynamic_escalation():
    ctrl = StepController(tau_step_confidence=-0.50)
    # Low confidence on small step causes escalation to large
    step = ctrl.evaluate_step_execution(
        step_index=2,
        prompt_text="Parse complex nested json",
        context_tokens=500,
        initial_tier="small",
        small_result={"min_logprob": -1.20, "cost_usd": 0.0, "latency_s": 0.20, "text": "invalid_syntax"},
        large_result={"cost_usd": 0.002, "latency_s": 0.70, "text": "{'status': 'ok'}"},
    )
    assert step.tier_chosen == "large"
    assert step.escalated is True
    assert step.wasted_latency_s == 0.20
    assert step.step_cost_usd == 0.002


def test_trajectory_aggregation():
    ctrl = StepController()
    s1 = AgentStep(
        step_index=1,
        prompt_text="Step 1",
        context_tokens=100,
        tier_chosen="small",
        output_text="ok",
        step_cost_usd=0.0,
        step_latency_s=0.15,
        escalated=False,
    )
    s2 = AgentStep(
        step_index=2,
        prompt_text="Step 2",
        context_tokens=300,
        tier_chosen="large",
        output_text="done",
        step_cost_usd=0.002,
        step_latency_s=0.65,
        escalated=True,
        wasted_cost_usd=0.0001,
        wasted_latency_s=0.15,
    )
    traj = ctrl.aggregate_trajectory(task_id="task-01", steps=[s1, s2], final_success=True)
    assert isinstance(traj, TrajectoryResult)
    assert traj.total_steps == 2
    assert traj.final_success is True
    assert traj.escalation_tax_ratio >= 1.0
