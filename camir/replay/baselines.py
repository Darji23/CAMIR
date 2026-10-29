"""Baseline and router policies for CAMIR replay evaluation (Phase 2).

Implements the eight baselines specified in Coding Plan v3 in order:
1. always-small: Routes every query to the local 1.5B tier.
2. always-medium: Routes every query to the local 7B tier.
3. always-large: Routes every query to the hosted API tier.
4. random-mix: Routes queries according to a fixed probability distribution.
5. oracle: Selects the lowest-cost tier that correctly answers the query.
6. logprob-cascade: Tries Small first; escalates to Large if logprob < threshold.
7. complexity-router: Heuristic routing inspired by LiteLLM documentation.
8. feature-router: Feature-based linear classifier on prompt structural signals.
"""

from abc import ABC, abstractmethod
import math
import random
import re
from typing import Any, Dict, Optional, Tuple
from camir.evaluation.scorers import score_prediction
from camir.traces.schema import TraceRow


class BasePolicy(ABC):
    """Abstract interface for all routing policies evaluated in replay."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the routing policy."""
        pass

    @abstractmethod
    def route(
        self,
        prompt_item: Dict[str, Any],
        traces: Dict[str, TraceRow],
    ) -> Tuple[str, TraceRow, Dict[str, Any]]:
        """Determine routing tier and return (chosen_tier, resulting_trace, metadata)."""
        pass


class AlwaysSmallPolicy(BasePolicy):
    @property
    def name(self) -> str:
        return "always-small"

    def route(self, prompt_item: Dict[str, Any], traces: Dict[str, TraceRow]):
        row = traces.get("small") or list(traces.values())[0]
        return "small", row, {"escalated": False, "tax_paid": 0.0}


class AlwaysMediumPolicy(BasePolicy):
    @property
    def name(self) -> str:
        return "always-medium"

    def route(self, prompt_item: Dict[str, Any], traces: Dict[str, TraceRow]):
        row = traces.get("medium") or traces.get("small") or list(traces.values())[0]
        return "medium", row, {"escalated": False, "tax_paid": 0.0}


class AlwaysLargePolicy(BasePolicy):
    @property
    def name(self) -> str:
        return "always-large"

    def route(self, prompt_item: Dict[str, Any], traces: Dict[str, TraceRow]):
        row = traces.get("large") or list(traces.values())[0]
        return "large", row, {"escalated": False, "tax_paid": 0.0}


class RandomMixPolicy(BasePolicy):
    """Randomly distributes traffic according to specified tier weights."""

    def __init__(self, p_small: float = 0.50, p_medium: float = 0.30, p_large: float = 0.20, seed: int = 42):
        self.p_small = p_small
        self.p_med = p_medium
        self.p_large = p_large
        self.rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "random-mix"

    def route(self, prompt_item: Dict[str, Any], traces: Dict[str, TraceRow]):
        val = self.rng.random()
        if val < self.p_small and "small" in traces:
            chosen = "small"
        elif val < (self.p_small + self.p_med) and "medium" in traces:
            chosen = "medium"
        else:
            chosen = "large" if "large" in traces else list(traces.keys())[0]

        row = traces.get(chosen) or list(traces.values())[0]
        return chosen, row, {"escalated": False, "tax_paid": 0.0}


class OraclePolicy(BasePolicy):
    """Cheapest tier that answers the query correctly. Upper bound on cost-accuracy."""

    @property
    def name(self) -> str:
        return "oracle"

    def route(self, prompt_item: Dict[str, Any], traces: Dict[str, TraceRow]):
        dataset = prompt_item["dataset"]
        ref = prompt_item.get("reference_answer", "")
        meta = prompt_item.get("metadata", {})

        # Order by cost: small ($0.00), medium ($0.00), large ($$$)
        ordered_tiers = ["small", "medium", "large"]
        for tier in ordered_tiers:
            if tier in traces:
                row = traces[tier]
                if score_prediction(dataset, row.output_text, ref, meta):
                    return tier, row, {"oracle_matched": True, "escalated": False}

        # If none correct, route to cheapest tier
        chosen = "small" if "small" in traces else list(traces.keys())[0]
        return chosen, traces[chosen], {"oracle_matched": False, "escalated": False}


class LogprobCascadePolicy(BasePolicy):
    """Cascading router: tries Small; escalates to Large if confidence < tau."""

    def __init__(self, threshold_min_lp: float = -0.65):
        self.tau = threshold_min_lp

    @property
    def name(self) -> str:
        return f"logprob-cascade (tau={self.tau})"

    def route(self, prompt_item: Dict[str, Any], traces: Dict[str, TraceRow]):
        row_s = traces.get("small")
        row_l = traces.get("large")

        if not row_s or not row_l:
            fallback = row_l or row_s or list(traces.values())[0]
            return "large", fallback, {"escalated": False}

        # Confidence signal: min token logprob or mean logprob
        conf = row_s.min_logprob if row_s.min_logprob is not None else (row_s.mean_logprob or -1.0)

        if conf >= self.tau:
            # Small tier accepted: zero escalation tax
            return "small", row_s, {"escalated": False, "confidence": conf}
        else:
            # Escalated to Large: incurs escalation tax (wasted small work)
            escalated_meta = {
                "escalated": True,
                "wasted_cost_usd": row_s.cost_usd,
                "wasted_latency_s": row_s.t_total_s,
                "wasted_energy_j": row_s.energy_j or 0.0,
                "confidence": conf,
            }
            return "large", row_l, escalated_meta


class ComplexityRouterPolicy(BasePolicy):
    """Rule-based routing heuristic inspired by LiteLLM router documentation.
    
    Routes to Large if prompt is long, contains code tokens, or math symbols;
    routes simple factual prompts to Small.
    """

    def __init__(self, char_length_threshold: int = 150):
        self.length_thresh = char_length_threshold

    @property
    def name(self) -> str:
        return "complexity-router (LiteLLM-style)"

    def route(self, prompt_item: Dict[str, Any], traces: Dict[str, TraceRow]):
        text = prompt_item["prompt"]
        
        # Complexity signals
        is_long = len(text) > self.length_thresh
        has_code = bool(re.search(r"\b(def|import|class|return|lambda|function)\b", text))
        has_math = bool(re.search(r"(\$|\\frac|\bcalculate\b|\bformula\b|\=|\bprobability\b)", text, re.IGNORECASE))

        if (is_long or has_code or has_math) and "large" in traces:
            chosen = "large"
        else:
            chosen = "small" if "small" in traces else list(traces.keys())[0]

        return chosen, traces[chosen], {"is_complex": (is_long or has_code or has_math)}


class FeatureRouterPolicy(BasePolicy):
    """Linear feature classifier predicting Small tier success probability."""

    def __init__(self, threshold: float = 0.50):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return f"feature-router (thresh={self.threshold})"

    def _extract_features(self, prompt: str) -> float:
        """Lightweight linear combination of prompt structural features."""
        length_score = -0.01 * len(prompt.split())
        digit_count = sum(c.isdigit() for c in prompt)
        math_score = -0.15 * digit_count
        punct_score = -0.05 * sum(prompt.count(p) for p in "?!;,")
        base_bias = 0.85
        logit = base_bias + length_score + math_score + punct_score
        # Sigmoid
        prob = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, logit))))
        return prob

    def route(self, prompt_item: Dict[str, Any], traces: Dict[str, TraceRow]):
        p_small_success = self._extract_features(prompt_item["prompt"])
        if p_small_success >= self.threshold and "small" in traces:
            chosen = "small"
        elif "large" in traces:
            chosen = "large"
        else:
            chosen = list(traces.keys())[0]

        return chosen, traces[chosen], {"predicted_success_prob": p_small_success}
