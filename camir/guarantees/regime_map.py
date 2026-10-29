"""Regime Map Decision Boundaries for CAMIR (Extension X2).

Analyzes the decision-theoretic space parameterized by:
- Cost ratio $\\rho = c_L / c_S$ (typically 10x to 100x)
- Escalation rate $e \\in [0, 1]$ (fraction of queries escalated from Small to Large)
- Verifier pass rate $v_{pass}$ (fraction of rejected small completions saved by the verifier)
- Verifier cost ratio $c_V / c_L$

Identifies the cost-minimal routing policy at matched quality:
- 'Always-Large': Optimal when quality cannot be compromised and escalation rate is extreme ($e > 1 - 1/\\rho$).
- 'Pure Cascade (P2)': Optimal when escalation is rare ($e < 0.20$) and Small success is high.
- 'Skip-Rule (P3)': Optimal when query difficulty is predictable and escalation is moderate.
- 'Verification Escalation (P5)': Dominates P2/P3 when verifier cost is low ($c_V / c_L < 0.20$) and $v_{pass} \\cdot c_L > c_V$.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class RegimePoint:
    cost_ratio_rho: float
    escalation_rate_e: float
    optimal_policy: str
    p2_cost: float
    p3_cost: float
    p5_cost: float
    always_large_cost: float
    net_savings_pct: float


def calculate_optimal_regime(
    cost_ratio_rho: float,
    escalation_rate_e: float,
    v_pass: float = 0.65,
    cv_to_cl_ratio: float = 0.10,
    p_skip: float = 0.25,
) -> RegimePoint:
    """Calculate the cost-minimal policy for a specific (rho, e) operating point.
    
    All costs normalized to c_S = 1.0 unit.
    Therefore c_L = rho.
    c_V = cv_to_cl_ratio * rho.
    """
    rho = max(1.1, cost_ratio_rho)
    e = max(0.01, min(0.99, escalation_rate_e))

    # Normalized costs
    c_s = 1.0
    c_l = rho
    c_v = cv_to_cl_ratio * rho

    # 1. Always Large Cost
    cost_always_large = c_l

    # 2. Pure Cascade (P2) Cost: small attempt + e * large attempt
    cost_p2 = c_s + e * c_l

    # 3. Skip-Rule (P3) Cost: skip fraction goes directly to large, remainder cascades
    cost_p3 = (1.0 - p_skip) * c_s + (1.0 - p_skip) * (e * 0.7) * c_l + p_skip * c_l

    # 4. Verification Escalation (P5) Cost: small attempt + e * verifier pass + e * (1 - v_pass) * large
    cost_p5 = c_s + e * c_v + e * (1.0 - v_pass) * c_l

    costs = {
        "Always-Large": cost_always_large,
        "P2-Cascade": cost_p2,
        "P3-SkipRule": cost_p3,
        "P5-Verification": cost_p5,
    }

    optimal_pol = min(costs, key=costs.get)
    min_cost = costs[optimal_pol]
    savings = max(0.0, (cost_always_large - min_cost) / cost_always_large) * 100.0

    return RegimePoint(
        cost_ratio_rho=round(rho, 2),
        escalation_rate_e=round(e, 4),
        optimal_policy=optimal_pol,
        p2_cost=round(cost_p2, 4),
        p3_cost=round(cost_p3, 4),
        p5_cost=round(cost_p5, 4),
        always_large_cost=round(cost_always_large, 4),
        net_savings_pct=round(savings, 2),
    )


def generate_regime_grid(
    rho_values: Optional[List[float]] = None,
    e_values: Optional[List[float]] = None,
    v_pass: float = 0.65,
    cv_to_cl_ratio: float = 0.10,
) -> List[RegimePoint]:
    """Generate 2D regime map grid over ranges of rho and escalation rate e."""
    if rho_values is None:
        rho_values = [5.0, 10.0, 20.0, 50.0, 100.0]
    if e_values is None:
        e_values = [0.10, 0.25, 0.50, 0.75, 0.90]

    grid = []
    for rho in rho_values:
        for e in e_values:
            pt = calculate_optimal_regime(
                cost_ratio_rho=rho,
                escalation_rate_e=e,
                v_pass=v_pass,
                cv_to_cl_ratio=cv_to_cl_ratio,
            )
            grid.append(pt)
    return grid
