"""Adaptive Online Recalibration for CAMIR (Extension X3).

Monitors empirical routing error under potential distribution shift
by auditing a random fraction gamma of live production traffic with the
Large tier, adjusting working conformal thresholds online and tracking audit costs.
"""

from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional, Tuple


@dataclass
class AuditRecord:
    """Record of an audited live inference request."""
    query_id: str
    timestamp_idx: int
    prompt_text: str
    chosen_tier: str
    small_correct: bool
    large_correct: bool
    loss: int
    audit_cost_usd: float


class AdaptiveRecalibrator:
    """Online audit controller and threshold adjuster under distribution drift."""

    def __init__(
        self,
        initial_tau: float = -0.50,
        target_alpha: float = 0.05,
        audit_rate_gamma: float = 0.10,
        window_size: int = 50,
        learning_rate: float = 0.05,
        seed: int = 42,
    ):
        self.current_tau = initial_tau
        self.target_alpha = target_alpha
        self.gamma = audit_rate_gamma
        self.window_size = window_size
        self.lr = learning_rate
        self.rng = random.Random(seed)

        self.audit_history: List[AuditRecord] = []
        self.total_queries_seen = 0
        self.total_audit_cost_usd = 0.0
        self.tau_history: List[Tuple[int, float]] = [(0, initial_tau)]

    def should_audit(self) -> bool:
        """Stochastically select whether to audit this query with the large tier."""
        return self.rng.random() < self.gamma

    def process_query(
        self,
        query_id: str,
        confidence: float,
        small_correct: bool,
        large_correct: bool,
        audit_cost_usd: float = 0.0002,
    ) -> Tuple[bool, float]:
        """Process incoming query, performing audit and online threshold update if selected.
        
        Returns:
            (was_audited: bool, current_tau: float)
        """
        self.total_queries_seen += 1
        is_audited = self.should_audit()

        if is_audited:
            accepted = (confidence >= self.current_tau)
            loss = 1 if (accepted and (not small_correct) and large_correct) else 0

            rec = AuditRecord(
                query_id=query_id,
                timestamp_idx=self.total_queries_seen,
                prompt_text="",
                chosen_tier="small" if accepted else "large",
                small_correct=small_correct,
                large_correct=large_correct,
                loss=loss,
                audit_cost_usd=audit_cost_usd,
            )
            self.audit_history.append(rec)
            self.total_audit_cost_usd += audit_cost_usd

            # Online threshold recalibration over moving audit window
            if len(self.audit_history) >= 10:
                recent_window = self.audit_history[-self.window_size:]
                empirical_risk = sum(r.loss for r in recent_window) / len(recent_window)

                # Gradient/proportional step:
                # If empirical risk > alpha: make tau higher (more conservative)
                # If empirical risk < alpha - 0.02: make tau lower (more permissive)
                error_diff = empirical_risk - self.target_alpha
                tau_step = self.lr * error_diff
                self.current_tau = max(-5.0, min(0.0, self.current_tau + tau_step))
                self.tau_history.append((self.total_queries_seen, round(self.current_tau, 4)))

        return is_audited, round(self.current_tau, 4)

    def get_audit_summary(self) -> Dict[str, any]:
        recent = self.audit_history[-self.window_size:] if self.audit_history else []
        emp_risk = (sum(r.loss for r in recent) / len(recent)) if recent else 0.0
        return {
            "total_queries_seen": self.total_queries_seen,
            "total_audits_performed": len(self.audit_history),
            "effective_audit_rate": round(len(self.audit_history) / max(1, self.total_queries_seen), 4),
            "total_audit_cost_usd": round(self.total_audit_cost_usd, 6),
            "current_working_tau": round(self.current_tau, 4),
            "recent_empirical_risk": round(emp_risk, 4),
            "target_alpha": self.target_alpha,
        }
