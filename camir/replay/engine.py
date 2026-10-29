"""Offline trace replay engine for CAMIR (Phase 2).

Replays routing policies over pre-collected immutable TraceRows, computing:
- Quality (Task accuracy scored against ground truth)
- Financial Cost ($ USD)
- High-resolution Latency: Mean, p50, p95, p99, TTFT, TPOT
- Package Energy in Joules
- Escalation Tax: Wasted latency/cost/energy on rejected small attempts

Strict timing and accounting invariant:
DOLLARS, SECONDS, AND JOULES ARE NEVER ADDED TOGETHER!
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.table import Table

from camir.evaluation.scorers import score_prediction
from camir.replay.baselines import BasePolicy
from camir.traces.schema import TraceRow
from camir.traces.store import TraceStore

console = Console()


@dataclass
class PolicyEvaluationResult:
    """Aggregated evaluation metrics for a single routing policy."""
    policy_name: str
    total_queries: int
    accuracy: float
    total_cost_usd: float
    avg_cost_usd: float
    mean_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    mean_ttft_s: float
    mean_tpot_s: float
    total_energy_j: float
    avg_energy_j: float
    tier_distribution: Dict[str, float]  # e.g. {"small": 0.65, "medium": 0.25, "large": 0.10}
    escalation_tax_ratio: float  # Total latency with tax / Direct optimal latency
    wasted_latency_s: float
    wasted_cost_usd: float
    per_source_accuracy: Dict[str, float] = field(default_factory=dict)


class ReplayEngine:
    """Executes offline trace replay across benchmark datasets."""

    def __init__(self, store: TraceStore):
        self.store = store

    def load_traces_for_prompts(
        self,
        prompt_items: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, TraceRow]]:
        """Load and index all stored traces for the given prompts by (prompt_id, tier)."""
        traces_by_prompt: Dict[str, Dict[str, TraceRow]] = {}
        with self.store._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM traces;")
            for r in cursor.fetchall():
                row_dict = dict(r)
                # Map integer booleans
                if "cache_disabled" in row_dict:
                    row_dict["cache_disabled"] = bool(row_dict["cache_disabled"])
                t_row = TraceRow(**row_dict)
                pid = t_row.prompt_id
                tier = t_row.tier.lower()
                traces_by_prompt.setdefault(pid, {})[tier] = t_row

        return traces_by_prompt

    def evaluate_policy(
        self,
        policy: BasePolicy,
        dataset_items: List[Dict[str, Any]],
        traces_by_prompt: Dict[str, Dict[str, TraceRow]],
    ) -> PolicyEvaluationResult:
        """Replay a single policy over the dataset items."""
        correct_count = 0
        total_cost = 0.0
        total_energy = 0.0
        latencies: List[float] = []
        ttfts: List[float] = []
        tpots: List[float] = []
        wasted_latency = 0.0
        wasted_cost = 0.0
        tier_counts = {"small": 0, "medium": 0, "large": 0}
        source_correct: Dict[str, int] = {}
        source_total: Dict[str, int] = {}

        evaluated_queries = 0

        for item in dataset_items:
            pid = item["prompt_id"]
            ds = item["dataset"]
            ref = item.get("reference_answer", "")
            meta = item.get("metadata", {})

            traces = traces_by_prompt.get(pid, {})
            if not traces:
                continue

            evaluated_queries += 1
            source_total[ds] = source_total.get(ds, 0) + 1

            # Route query via policy
            chosen_tier, chosen_trace, route_meta = policy.route(item, traces)
            tier_counts[chosen_tier] = tier_counts.get(chosen_tier, 0) + 1

            # Determine task correctness
            is_correct = score_prediction(ds, chosen_trace.output_text, ref, meta)
            if is_correct:
                correct_count += 1
                source_correct[ds] = source_correct.get(ds, 0) + 1

            # Compute effective cost and latency including escalation tax
            effective_latency = chosen_trace.t_total_s
            effective_cost = chosen_trace.cost_usd
            effective_energy = chosen_trace.energy_j or 0.0

            if route_meta.get("escalated"):
                w_lat = route_meta.get("wasted_latency_s", 0.0)
                w_c = route_meta.get("wasted_cost_usd", 0.0)
                w_e = route_meta.get("wasted_energy_j", 0.0)
                effective_latency += w_lat
                effective_cost += w_c
                effective_energy += w_e
                wasted_latency += w_lat
                wasted_cost += w_c

            latencies.append(effective_latency)
            ttfts.append(chosen_trace.t_ttft_s)
            tpots.append(chosen_trace.tpot_s)
            total_cost += effective_cost
            total_energy += effective_energy

        n = max(1, evaluated_queries)
        acc = correct_count / n
        lat_arr = np.array(latencies) if latencies else np.array([0.0])

        per_source_acc = {
            s: round(source_correct.get(s, 0) / max(1, source_total.get(s, 0)), 4)
            for s in source_total
        }

        tier_dist = {
            t: round(tier_counts.get(t, 0) / n, 4)
            for t in ["small", "medium", "large"]
        }

        # Escalation tax ratio: total latency / (total latency - wasted latency)
        net_lat = sum(latencies)
        ideal_lat = max(1e-6, net_lat - wasted_latency)
        tax_ratio = round(net_lat / ideal_lat, 4)

        return PolicyEvaluationResult(
            policy_name=policy.name,
            total_queries=evaluated_queries,
            accuracy=round(acc, 4),
            total_cost_usd=round(total_cost, 6),
            avg_cost_usd=round(total_cost / n, 8),
            mean_latency_s=round(float(np.mean(lat_arr)), 4),
            p50_latency_s=round(float(np.percentile(lat_arr, 50)), 4),
            p95_latency_s=round(float(np.percentile(lat_arr, 95)), 4),
            p99_latency_s=round(float(np.percentile(lat_arr, 99)), 4),
            mean_ttft_s=round(float(np.mean(ttfts)) if ttfts else 0.0, 4),
            mean_tpot_s=round(float(np.mean(tpots)) if tpots else 0.0, 4),
            total_energy_j=round(total_energy, 4),
            avg_energy_j=round(total_energy / n, 4),
            tier_distribution=tier_dist,
            escalation_tax_ratio=tax_ratio,
            wasted_latency_s=round(wasted_latency, 4),
            wasted_cost_usd=round(wasted_cost, 6),
            per_source_accuracy=per_source_acc,
        )

    def print_frontier_table(self, results: List[PolicyEvaluationResult]):
        """Render the Policy Frontier comparison table across baselines."""
        table = Table(title="CAMIR Replay Frontier: Baselines & Routing Policies", show_header=True)
        table.add_column("Policy", style="cyan", no_wrap=True)
        table.add_column("Accuracy", justify="right", style="bold green")
        table.add_column("Avg Cost ($)", justify="right", style="yellow")
        table.add_column("p50 Lat (ms)", justify="right")
        table.add_column("p95 Lat (ms)", justify="right")
        table.add_column("Energy (J)", justify="right")
        table.add_column("Tier Dist (S/M/L)", justify="center")
        table.add_column("Tax Ratio", justify="right", style="red")

        for r in results:
            d = r.tier_distribution
            dist_str = f"{int(d.get('small',0)*100)}% / {int(d.get('medium',0)*100)}% / {int(d.get('large',0)*100)}%"
            table.add_row(
                r.policy_name,
                f"{r.accuracy * 100:.1f}%",
                f"${r.avg_cost_usd:.6f}",
                f"{r.p50_latency_s * 1000:.1f}",
                f"{r.p95_latency_s * 1000:.1f}",
                f"{r.avg_energy_j:.2f} J",
                dist_str,
                f"{r.escalation_tax_ratio:.2f}x",
            )
        console.print(table)
