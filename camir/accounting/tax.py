"""Escalation Tax Accounting for CAMIR (Contribution C1).

In confidence-gated cascades and multi-tier routing, when a small model's
attempt is rejected by a gate or verifier and re-routed to the large tier:
- The small model's generation is wasted work.
- The gate inspection is wasted overhead.
- The verifier scoring is auxiliary overhead.

Escalation Tax ($W$):
    $W_{lat} = t_S + t_G + t_V$
    $W_{cost} = c_S + c_G + c_V$
    $W_{energy} = E_S + E_G + E_V$

Tax Multiplier ($T$):
    $T = Actual Work / Direct Work$ (where Direct Work is the work if the large tier was called directly)

STRICT ACCOUNTING INVARIANT:
DOLLARS, SECONDS, AND JOULES ARE NEVER ADDED TOGETHER!
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class EscalationTaxRecord:
    """Detailed attribution of useful and wasted work for a single query."""
    prompt_id: str
    policy: str
    escalated: bool
    final_tier: str

    # Direct baseline work (ideal direct routing to final tier)
    direct_cost_usd: float
    direct_latency_s: float
    direct_energy_j: float

    # Actual total work incurred
    actual_cost_usd: float
    actual_latency_s: float
    actual_energy_j: float

    # Wasted overhead (small attempt discarded + verification passes)
    wasted_cost_usd: float = 0.0
    wasted_latency_s: float = 0.0
    wasted_energy_j: float = 0.0

    @property
    def useful_cost_usd(self) -> float:
        return round(max(0.0, self.actual_cost_usd - self.wasted_cost_usd), 8)

    @property
    def useful_latency_s(self) -> float:
        return round(max(0.0, self.actual_latency_s - self.wasted_latency_s), 6)

    @property
    def useful_energy_j(self) -> float:
        return round(max(0.0, self.actual_energy_j - self.wasted_energy_j), 6)

    @property
    def latency_tax_multiplier(self) -> float:
        """Ratio of total time to direct time: actual / direct."""
        if self.direct_latency_s <= 1e-6:
            return 1.0
        return round(self.actual_latency_s / self.direct_latency_s, 4)

    @property
    def cost_tax_multiplier(self) -> float:
        """Ratio of total cost to direct cost: actual / direct."""
        if self.direct_cost_usd <= 1e-8:
            return 1.0
        return round(self.actual_cost_usd / self.direct_cost_usd, 4)

    @property
    def energy_tax_multiplier(self) -> float:
        """Ratio of total energy to direct energy: actual / direct."""
        if self.direct_energy_j <= 1e-6:
            return 1.0
        return round(self.actual_energy_j / self.direct_energy_j, 4)


@dataclass
class EscalationTaxSummary:
    """Aggregated tax metrics across an entire evaluation corpus."""
    policy: str
    total_queries: int
    escalation_count: int
    escalation_rate: float
    total_wasted_latency_s: float
    total_wasted_cost_usd: float
    total_wasted_energy_j: float
    avg_latency_tax_multiplier: float
    avg_cost_tax_multiplier: float
    avg_energy_tax_multiplier: float
    p50_latency_s: float
    p95_latency_s: float


class EscalationAccountant:
    """Accountant collecting and attributing useful vs wasted work across queries."""

    def __init__(self, policy_name: str = "cascade"):
        self.policy_name = policy_name
        self.records: List[EscalationTaxRecord] = []

    def record_query(
        self,
        prompt_id: str,
        escalated: bool,
        final_tier: str,
        direct_latency_s: float,
        direct_cost_usd: float,
        direct_energy_j: float,
        actual_latency_s: float,
        actual_cost_usd: float,
        actual_energy_j: float,
        wasted_latency_s: float = 0.0,
        wasted_cost_usd: float = 0.0,
        wasted_energy_j: float = 0.0,
    ) -> EscalationTaxRecord:
        """Record and validate query work breakdown."""
        # Sanity validation
        assert actual_latency_s >= wasted_latency_s - 1e-6, "Wasted latency cannot exceed actual latency!"
        assert actual_cost_usd >= wasted_cost_usd - 1e-8, "Wasted cost cannot exceed actual cost!"
        assert actual_energy_j >= wasted_energy_j - 1e-6, "Wasted energy cannot exceed actual energy!"

        rec = EscalationTaxRecord(
            prompt_id=prompt_id,
            policy=self.policy_name,
            escalated=escalated,
            final_tier=final_tier,
            direct_cost_usd=direct_cost_usd,
            direct_latency_s=direct_latency_s,
            direct_energy_j=direct_energy_j,
            actual_cost_usd=actual_cost_usd,
            actual_latency_s=actual_latency_s,
            actual_energy_j=actual_energy_j,
            wasted_cost_usd=wasted_cost_usd,
            wasted_latency_s=wasted_latency_s,
            wasted_energy_j=wasted_energy_j,
        )
        self.records.append(rec)
        return rec

    def compute_summary(self) -> EscalationTaxSummary:
        """Compute macro-averaged tax statistics."""
        n = max(1, len(self.records))
        n_esc = sum(1 for r in self.records if r.escalated)
        esc_rate = n_esc / n

        tot_w_lat = sum(r.wasted_latency_s for r in self.records)
        tot_w_cost = sum(r.wasted_cost_usd for r in self.records)
        tot_w_energy = sum(r.wasted_energy_j for r in self.records)

        lat_multipliers = [r.latency_tax_multiplier for r in self.records]
        cost_multipliers = [r.cost_tax_multiplier for r in self.records]
        energy_multipliers = [r.energy_tax_multiplier for r in self.records]

        actual_latencies = [r.actual_latency_s for r in self.records]

        return EscalationTaxSummary(
            policy=self.policy_name,
            total_queries=n,
            escalation_count=n_esc,
            escalation_rate=round(esc_rate, 4),
            total_wasted_latency_s=round(tot_w_lat, 4),
            total_wasted_cost_usd=round(tot_w_cost, 6),
            total_wasted_energy_j=round(tot_w_energy, 4),
            avg_latency_tax_multiplier=round(float(np.mean(lat_multipliers)), 4),
            avg_cost_tax_multiplier=round(float(np.mean(cost_multipliers)), 4),
            avg_energy_tax_multiplier=round(float(np.mean(energy_multipliers)), 4),
            p50_latency_s=round(float(np.percentile(actual_latencies, 50)), 4) if actual_latencies else 0.0,
            p95_latency_s=round(float(np.percentile(actual_latencies, 95)), 4) if actual_latencies else 0.0,
        )
