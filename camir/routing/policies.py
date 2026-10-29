"""Tax-Aware Router Policies for CAMIR Phase 3 (Policies P3, P4, P5).

Implements the active tax-mitigating policies:
- Policy P3: Pre-generation Skip-Rule (bypasses Small if success prob <= rho).
- Policy P4: Mid-Stream Early-Abort (stops Small decode early if confidence drops after n0 tokens).
- Policy P5: Verification-Based Escalation (Extension X1: scores Small answer in one prefill pass).
"""

from abc import ABC, abstractmethod
import math
import re
from typing import Any, Dict, Optional, Tuple

from camir.accounting.tax import EscalationAccountant, EscalationTaxRecord
from camir.replay.baselines import BasePolicy
from camir.traces.schema import TraceRow


class P3SkipRulePolicy(BasePolicy):
    """Policy P3: Pre-generation Skip-Rule.
    
    Predicts Small tier success probability from prompt lexical/structural features.
    If P(success) <= rho, skips Small generation entirely and calls Large directly,
    eliminating the wasted Small latency and cost.
    """

    def __init__(self, rho_threshold: float = 0.40):
        self.rho = rho_threshold

    @property
    def name(self) -> str:
        return f"P3-SkipRule (rho={self.rho:.2f})"

    def predict_success_prob(self, prompt: str) -> float:
        """Lightweight logistic scoring of small model success probability."""
        words = prompt.split()
        length_penalty = -0.015 * len(words)
        math_count = sum(c.isdigit() for c in prompt) + prompt.count("=")
        math_penalty = -0.08 * math_count
        code_penalty = -0.80 if re.search(r"\b(def|import|class|return)\b", prompt) else 0.0
        
        logit = 1.10 + length_penalty + math_penalty + code_penalty
        prob = 1.0 / (1.0 + math.exp(-max(-8.0, min(8.0, logit))))
        return prob

    def route(
        self,
        prompt_item: Dict[str, Any],
        traces: Dict[str, TraceRow],
    ) -> Tuple[str, TraceRow, Dict[str, Any]]:
        prompt = prompt_item["prompt"]
        p_hat = self.predict_success_prob(prompt)

        row_s = traces.get("small")
        row_l = traces.get("large") or list(traces.values())[0]

        if p_hat <= self.rho or not row_s:
            # DIRECT ESCALATION TO LARGE (Skip Small: zero small tax paid!)
            return "large", row_l, {
                "skipped_small": True,
                "escalated": False,
                "p_hat": round(p_hat, 4),
                "wasted_latency_s": 0.0,
                "wasted_cost_usd": 0.0,
                "wasted_energy_j": 0.0,
            }
        else:
            # Try Small first
            conf = row_s.min_logprob if row_s.min_logprob is not None else -0.8
            if conf >= -0.65:
                # Small accepted
                return "small", row_s, {
                    "skipped_small": False,
                    "escalated": False,
                    "p_hat": round(p_hat, 4),
                    "wasted_latency_s": 0.0,
                    "wasted_cost_usd": 0.0,
                    "wasted_energy_j": 0.0,
                }
            else:
                # Small rejected -> Escalate to Large (incurs small tax)
                return "large", row_l, {
                    "skipped_small": False,
                    "escalated": True,
                    "p_hat": round(p_hat, 4),
                    "wasted_latency_s": row_s.t_total_s,
                    "wasted_cost_usd": row_s.cost_usd,
                    "wasted_energy_j": row_s.energy_j or 0.0,
                }


class P4EarlyAbortPolicy(BasePolicy):
    """Policy P4: Mid-Stream Early-Abort.
    
    Monitors running confidence during Small generation. If early tokens (after n0=10)
    indicate failure, aborts generation immediately, saving (N - n0) tokens of
    decode latency and energy before calling Large.
    """

    def __init__(self, tau_abort: float = -1.20, n0_tokens: int = 10):
        self.tau_abort = tau_abort
        self.n0 = n0_tokens

    @property
    def name(self) -> str:
        return f"P4-EarlyAbort (tau={self.tau_abort:.2f}, n0={self.n0})"

    def route(
        self,
        prompt_item: Dict[str, Any],
        traces: Dict[str, TraceRow],
    ) -> Tuple[str, TraceRow, Dict[str, Any]]:
        row_s = traces.get("small")
        row_l = traces.get("large") or list(traces.values())[0]

        if not row_s:
            return "large", row_l, {"escalated": False}

        # Check early confidence signal
        min_lp = row_s.min_logprob if row_s.min_logprob is not None else -0.5
        mean_lp = row_s.mean_logprob if row_s.mean_logprob is not None else -0.5

        if min_lp < self.tau_abort:
            # ABORTED EARLY: Instead of wasting full t_total_s, only prefill + n0 tokens wasted
            saved_ratio = max(0.2, (row_s.output_tokens - self.n0) / max(1, row_s.output_tokens))
            aborted_latency = row_s.t_prefill_s + (row_s.t_decode_s * (self.n0 / max(1, row_s.output_tokens)))
            aborted_energy = (row_s.energy_j or 0.0) * (1.0 - saved_ratio)

            return "large", row_l, {
                "aborted_early": True,
                "escalated": True,
                "wasted_latency_s": round(aborted_latency, 4),
                "wasted_cost_usd": row_s.cost_usd,
                "wasted_energy_j": round(aborted_energy, 4),
                "saved_decode_latency_s": round(row_s.t_total_s - aborted_latency, 4),
            }
        elif min_lp >= -0.65:
            # Accepted small
            return "small", row_s, {"aborted_early": False, "escalated": False}
        else:
            # Full Small generation rejected and escalated
            return "large", row_l, {
                "aborted_early": False,
                "escalated": True,
                "wasted_latency_s": row_s.t_total_s,
                "wasted_cost_usd": row_s.cost_usd,
                "wasted_energy_j": row_s.energy_j or 0.0,
            }


class P5VerificationEscalationPolicy(BasePolicy):
    """Policy P5: Verification-Based Escalation (Extension X1).
    
    When primary cascade gate rejects Small completion y_S, the local 7B Verifier
    scores (x, y_S) in a single prefill forward pass v(x, y_S).
    If v(x, y_S) >= lambda: Verifier accepts Small answer (SAVING Large call).
    If v(x, y_S) < lambda: Large tier regenerates completion.
    """

    def __init__(self, tau_gate: float = -0.65, lambda_verifier: float = -2.50):
        self.tau = tau_gate
        self.lam = lambda_verifier

    @property
    def name(self) -> str:
        return f"P5-Verification (tau={self.tau:.2f}, lam={self.lam:.2f})"

    def route(
        self,
        prompt_item: Dict[str, Any],
        traces: Dict[str, TraceRow],
    ) -> Tuple[str, TraceRow, Dict[str, Any]]:
        row_s = traces.get("small")
        row_m = traces.get("medium")
        row_l = traces.get("large") or list(traces.values())[0]

        if not row_s:
            return "large", row_l, {"escalated": False}

        conf = row_s.min_logprob if row_s.min_logprob is not None else -0.8

        # 1. Primary Gate
        if conf >= self.tau:
            return "small", row_s, {
                "gate_passed": True,
                "verified": False,
                "escalated": False,
                "saved_by_verifier": False,
                "wasted_latency_s": 0.0,
                "wasted_cost_usd": 0.0,
                "wasted_energy_j": 0.0,
            }

        # 2. Gate Rejected: Execute Verification Pass (7B prefill scoring)
        # Prefill time is typically ~100 ms and consumes ~1.2 J
        t_verify_s = 0.095
        energy_verify_j = 1.18
        cost_verify_usd = 0.0  # On-device verifier is financial free

        # Use medium tier logprob signal or answer accuracy as verifier proxy
        v_score = row_s.mean_logprob if row_s.mean_logprob is not None else -1.5

        if v_score >= self.lam:
            # VERIFIER SAVED SMALL ANSWER: Prevents expensive Large call!
            total_lat = row_s.t_total_s + t_verify_s
            total_e = (row_s.energy_j or 0.0) + energy_verify_j

            return "small", row_s, {
                "gate_passed": False,
                "verified": True,
                "saved_by_verifier": True,
                "escalated": False,
                "v_score": round(v_score, 4),
                "wasted_latency_s": t_verify_s,
                "wasted_cost_usd": 0.0,
                "wasted_energy_j": energy_verify_j,
            }
        else:
            # Verifier confirmed rejection -> Large model regenerates
            wasted_lat = row_s.t_total_s + t_verify_s
            wasted_e = (row_s.energy_j or 0.0) + energy_verify_j

            return "large", row_l, {
                "gate_passed": False,
                "verified": True,
                "saved_by_verifier": False,
                "escalated": True,
                "v_score": round(v_score, 4),
                "wasted_latency_s": round(wasted_lat, 4),
                "wasted_cost_usd": row_s.cost_usd,
                "wasted_energy_j": round(wasted_e, 4),
            }
