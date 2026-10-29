"""Multi-Step Agentic Routing & StepController (Contribution C4 / Experiment E9).

Manages per-step dynamic tier allocation for multi-turn agentic workflows (e.g. BFCL, tool use, code refinement).
Tracks context-length growth, per-step escalation tax, and cumulative trajectory costs.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class AgentStep(BaseModel):
    """Execution record for a single reasoning hop or tool-use turn."""
    step_index: int
    prompt_text: str
    context_tokens: int
    tier_chosen: str
    output_text: str
    step_cost_usd: float
    step_latency_s: float
    escalated: bool = False
    wasted_cost_usd: float = 0.0
    wasted_latency_s: float = 0.0


class TrajectoryResult(BaseModel):
    """Aggregate result for a multi-turn agentic task trajectory."""
    task_id: str
    total_steps: int
    final_success: bool
    total_cost_usd: float
    total_latency_s: float
    wasted_cost_usd: float
    wasted_latency_s: float
    escalation_tax_ratio: float
    steps: List[AgentStep] = Field(default_factory=list)


class StepController:
    """Dynamic per-step router for multi-turn agentic workflows."""

    def __init__(
        self,
        tau_step_confidence: float = -0.55,
        max_small_context_tokens: int = 1500,
        cost_ratio_rho: float = 20.0,
    ):
        self.tau_step_confidence = tau_step_confidence
        self.max_small_context_tokens = max_small_context_tokens
        self.cost_ratio_rho = cost_ratio_rho

    def decide_step_tier(
        self,
        step_index: int,
        context_tokens: int,
        predicted_step_complexity: float = 0.0,
    ) -> str:
        """Decide tier for the upcoming step before execution.
        
        If context length exceeds small tier threshold or step complexity is high,
        skip small and route directly to Large tier.
        """
        if context_tokens > self.max_small_context_tokens:
            return "large"
        if predicted_step_complexity > 0.80:
            return "large"
        return "small"

    def evaluate_step_execution(
        self,
        step_index: int,
        prompt_text: str,
        context_tokens: int,
        initial_tier: str,
        small_result: Optional[Dict[str, Any]] = None,
        large_result: Optional[Dict[str, Any]] = None,
    ) -> AgentStep:
        """Evaluate a step's execution and handle dynamic escalation if required."""
        if initial_tier == "large" or small_result is None:
            # Direct to large tier
            l_res = large_result or {}
            return AgentStep(
                step_index=step_index,
                prompt_text=prompt_text,
                context_tokens=context_tokens,
                tier_chosen="large",
                output_text=l_res.get("text", ""),
                step_cost_usd=l_res.get("cost_usd", 0.0),
                step_latency_s=l_res.get("latency_s", 0.0),
                escalated=False,
            )

        # Evaluated on small tier first
        s_min_lp = small_result.get("min_logprob", 0.0)
        s_cost = small_result.get("cost_usd", 0.0)
        s_lat = small_result.get("latency_s", 0.0)
        s_text = small_result.get("text", "")

        # Check if small tier execution passes confidence threshold
        if s_min_lp >= self.tau_step_confidence and not small_result.get("tool_syntax_error", False):
            # Accepted small step
            return AgentStep(
                step_index=step_index,
                prompt_text=prompt_text,
                context_tokens=context_tokens,
                tier_chosen="small",
                output_text=s_text,
                step_cost_usd=s_cost,
                step_latency_s=s_lat,
                escalated=False,
            )

        # Confidence too low or syntax error: escalate step to Large tier
        l_res = large_result or {}
        l_cost = l_res.get("cost_usd", 0.0)
        l_lat = l_res.get("latency_s", 0.0)
        l_text = l_res.get("text", "")

        return AgentStep(
            step_index=step_index,
            prompt_text=prompt_text,
            context_tokens=context_tokens,
            tier_chosen="large",
            output_text=l_text,
            step_cost_usd=s_cost + l_cost,
            step_latency_s=s_lat + l_lat,
            escalated=True,
            wasted_cost_usd=s_cost,
            wasted_latency_s=s_lat,
        )

    def aggregate_trajectory(
        self,
        task_id: str,
        steps: List[AgentStep],
        final_success: bool,
    ) -> TrajectoryResult:
        """Aggregate step-wise records into a complete trajectory evaluation."""
        total_cost = sum(s.step_cost_usd for s in steps)
        total_lat = sum(s.step_latency_s for s in steps)
        wasted_cost = sum(s.wasted_cost_usd for s in steps)
        wasted_lat = sum(s.wasted_latency_s for s in steps)

        # Escalation tax ratio: total cost relative to useful work
        useful_cost = max(0.000001, total_cost - wasted_cost)
        tax_ratio = total_cost / useful_cost

        return TrajectoryResult(
            task_id=task_id,
            total_steps=len(steps),
            final_success=final_success,
            total_cost_usd=round(total_cost, 6),
            total_latency_s=round(total_lat, 4),
            wasted_cost_usd=round(wasted_cost, 6),
            wasted_latency_s=round(wasted_lat, 4),
            escalation_tax_ratio=round(tax_ratio, 2),
            steps=steps,
        )
