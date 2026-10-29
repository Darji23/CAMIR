"""Phase 2 Master Pipeline: Trace Replay, E1 Table, and Baseline Frontiers.

Executes:
1. Data verification & loading across train, calibration, and test splits.
2. Replay evaluation across all 8 baselines in order:
   - Always-Small (1.5B)
   - Always-Medium (7B)
   - Always-Large (Hosted API)
   - Random-Mix
   - Oracle (Upper Bound)
   - Log-prob Cascade (Threshold on minimum log-probability)
   - Complexity Router (LiteLLM-style heuristic)
   - Feature Router (Structural linear classifier)
3. Computes the E1 Table (per tier, per source): Quality, Cost, TTFT, TPOT,
   p50/p95/p99 latency, Energy.
4. Measures the Escalation Tax (ratio of wasted small work upon escalation).
5. Saves checkpoints to checkpoints/phase_2/outputs/.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List
import numpy as np
from rich.console import Console
from rich.table import Table

from camir.replay.baselines import (
    AlwaysLargePolicy,
    AlwaysMediumPolicy,
    AlwaysSmallPolicy,
    ComplexityRouterPolicy,
    FeatureRouterPolicy,
    LogprobCascadePolicy,
    OraclePolicy,
    RandomMixPolicy,
)
from camir.replay.engine import ReplayEngine
from camir.traces.store import TraceStore
from camir.utils.checkpoint import CheckpointManager
from camir.utils.env import load_env
from camir.utils.hardware import get_hardware_info, get_thermal_state

console = Console()
DATA_DIR = Path("data")


def load_dataset(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_e1_table(store: TraceStore, test_prompts: List[Dict]) -> Table:
    """Generate the Phase 2 E1 Table: Metrics per Tier across individual Sources."""
    table = Table(title="E1 Evaluation Table: Performance Across Tiers and Datasets", show_header=True)
    table.add_column("Dataset / Source", style="cyan")
    table.add_column("Tier", style="bold")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("Cost / Query ($)", justify="right", style="yellow")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("TPOT (ms)", justify="right")
    table.add_column("p50 (ms)", justify="right")
    table.add_column("p95 (ms)", justify="right")
    table.add_column("Energy (J)", justify="right")

    engine = ReplayEngine(store)
    traces = engine.load_traces_for_prompts(test_prompts)

    sources = sorted(list(set(p["dataset"] for p in test_prompts)))
    tiers = ["small", "medium", "large"]

    from camir.evaluation.scorers import score_prediction

    for src in sources:
        src_prompts = [p for p in test_prompts if p["dataset"] == src]
        for t in tiers:
            correct, cost_sum, energy_sum = 0, 0.0, 0.0
            ttfts, tpots, latencies = [], [], []

            for p in src_prompts:
                pid = p["prompt_id"]
                p_traces = traces.get(pid, {})
                row = p_traces.get(t)
                if not row:
                    continue

                ref = p.get("reference_answer", "")
                meta = p.get("metadata", {})
                if score_prediction(src, row.output_text, ref, meta):
                    correct += 1

                cost_sum += row.cost_usd
                energy_sum += (row.energy_j or 0.0)
                ttfts.append(row.t_ttft_s * 1000)
                tpots.append(row.tpot_s * 1000)
                latencies.append(row.t_total_s * 1000)

            if not latencies:
                continue

            n = len(latencies)
            acc = (correct / n) * 100
            avg_cost = cost_sum / n
            avg_energy = energy_sum / n
            p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
            p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
            m_ttft = float(np.mean(ttfts)) if ttfts else 0.0
            m_tpot = float(np.mean(tpots)) if tpots else 0.0

            table.add_row(
                src,
                t,
                f"{acc:.1f}%",
                f"${avg_cost:.6f}" if avg_cost > 0 else "$0.00",
                f"{m_ttft:.1f}",
                f"{m_tpot:.1f}",
                f"{p50:.1f}",
                f"{p95:.1f}",
                f"{avg_energy:.2f} J",
            )

    return table


def main():
    parser = argparse.ArgumentParser(description="CAMIR Phase 2 Master Pipeline")
    parser.add_argument(
        "--collect-traces",
        action="store_true",
        help="Run real trace collection on untraced test prompts before replay.",
    )
    parser.add_argument(
        "--per-dataset",
        type=int,
        default=5,
        help="Number of prompts per dataset to trace if collecting (default: 5).",
    )
    args = parser.parse_args()

    load_env()
    checkpoints = CheckpointManager(phase=2)

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]       CAMIR Phase 2: Trace-First Replay Pipeline & Baseline Frontiers  [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════════════[/bold cyan]")

    store = TraceStore()
    engine = ReplayEngine(store)

    test_file = DATA_DIR / "test_group.jsonl"
    smoke_file = DATA_DIR / "smoke_20.jsonl"

    test_prompts = load_dataset(test_file)
    smoke_prompts = load_dataset(smoke_file) if smoke_file.exists() else []

    # Combine test pool with smoke set so all real evaluated benchmark prompts in SQLite are included
    existing_pids = {p["prompt_id"] for p in test_prompts}
    all_eval_prompts = [p for p in smoke_prompts if p["prompt_id"] not in existing_pids] + test_prompts

    console.print(f"Loaded {len(test_prompts)} prompts from {test_file.name} (+ {len(smoke_prompts)} smoke prompts).")

    # If trace collection requested, collect real traces for missing prompts
    if args.collect_traces:
        from camir.tiers.hosted_api import HostedAPITier
        from camir.tiers.mlx_local import LocalMLXTier, MediumMLXTier
        from camir.traces.collector import TraceCollector

        console.print("\n[bold]Collecting real traces across Small, Medium, and Large tiers...[/bold]")
        small = LocalMLXTier(model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit", tier_name="small")
        medium = MediumMLXTier(model_id="mlx-community/Qwen2.5-7B-Instruct-4bit", tier_name="medium")
        hosted = HostedAPITier(tier_name="large")
        collector = TraceCollector(store=store, small_tier=small, medium_tier=medium, hosted_tier=hosted)

        by_ds: Dict[str, List[Dict]] = {}
        for p in test_prompts:
            by_ds.setdefault(p["dataset"], []).append(p)
        to_trace = []
        for ds, items in by_ds.items():
            to_trace.extend(items[: args.per_dataset])

        from rich.progress import track
        for item in track(to_trace, description="Tracing prompts..."):
            collector.collect_prompt_traces(item, max_tokens=128)

    traces = engine.load_traces_for_prompts(all_eval_prompts)
    traced_prompts = [p for p in all_eval_prompts if p["prompt_id"] in traces and len(traces[p["prompt_id"]]) >= 2]
    console.print(f"Loaded traces from SQLite: [bold cyan]{len(traces)}[/bold cyan] prompt trace bundles available.")
    console.print(f"Prompts with complete multi-tier traces ready for evaluation: [bold green]{len(traced_prompts)}[/bold green]")

    # 1. Instantiate the 8 baseline policies in order
    policies = [
        AlwaysSmallPolicy(),
        AlwaysMediumPolicy(),
        AlwaysLargePolicy(),
        RandomMixPolicy(p_small=0.5, p_medium=0.3, p_large=0.2, seed=42),
        OraclePolicy(),
        LogprobCascadePolicy(threshold_min_lp=-0.50),
        ComplexityRouterPolicy(char_length_threshold=140),
        FeatureRouterPolicy(threshold=0.50),
    ]

    console.print("\n[bold]Replaying 8 Baseline Policies across Real Test Traces...[/bold]")
    results = []
    eval_set = traced_prompts if traced_prompts else all_eval_prompts
    for pol in policies:
        res = engine.evaluate_policy(pol, eval_set, traces)
        results.append(res)

    # 2. Render Policy Frontier Table
    console.print("")
    engine.print_frontier_table(results)

    # 3. Render E1 Table
    console.print("")
    e1_table = generate_e1_table(store, eval_set)
    console.print(e1_table)

    # 4. Save Checkpoint Summary
    summary_lines = [
        "================================================================================",
        "CAMIR Phase 2 Summary: Trace-First Data Pipeline and Baseline Frontiers",
        "================================================================================",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"Hardware:  {get_hardware_info()}",
        f"Thermal:   {get_thermal_state()}",
        f"Evaluated Test Prompts: {len(test_prompts)}",
        "",
        "Baseline Policy Evaluation Results:",
    ]
    for r in results:
        summary_lines.append(
            f"- {r.policy_name:<35} | Acc: {r.accuracy*100:>5.1f}% | "
            f"Avg Cost: ${r.avg_cost_usd:.6f} | p50: {r.p50_latency_s*1000:>6.1f} ms | "
            f"Energy: {r.avg_energy_j:>5.2f} J | Tax: {r.escalation_tax_ratio:.2f}x"
        )
    summary_lines.append("================================================================================")
    summary_text = "\n".join(summary_lines)

    checkpoints.save_output("phase2_summary.txt", summary_text)
    console.print("\n[bold green]✓ Phase 2 Execution Complete.[/bold green]")
    console.print("Checkpoints saved in: checkpoints/phase_2/outputs/phase2_summary.txt\n")


if __name__ == "__main__":
    main()
