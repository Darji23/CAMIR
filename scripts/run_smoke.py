"""End-to-End Phase 1 Smoke Pipeline Driver.

Runs the 20-prompt smoke dataset through the Small Tier (local MLX)
and Large Tier (hosted API), saves validated TraceRows to SQLite,
and exports to Parquet.

Execution:
    python scripts/run_smoke.py
    python scripts/run_smoke.py --mock-local   # fast dry-run without downloading MLX weights
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from rich.console import Console
from rich.progress import track
from rich.table import Table

from camir.accounting.cost import calculate_token_cost
from camir.tiers.hosted_api import HostedAPITier
from camir.tiers.mlx_local import LocalMLXTier
from camir.traces.schema import TimingBreakdown, TraceRow
from camir.traces.store import TraceStore
from camir.utils.hardware import get_hardware_info, get_thermal_state
from camir.utils.hashing import compute_config_hash

console = Console()
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SMOKE_FILE = DATA_DIR / "smoke_20.jsonl"
TRACES_DIR = Path(__file__).resolve().parent.parent / "traces"


def run_smoke_pipeline(mock_local: bool = False, max_prompts: int = 20):
    console.rule("[bold cyan]CAMIR Phase 1: 20-Prompt Smoke Pipeline[/bold cyan]")

    if not SMOKE_FILE.exists():
        console.print(f"[bold red]Error: Smoke dataset not found at {SMOKE_FILE}.[/bold red]")
        console.print("Run: python camir/data/download_and_prep.py first.")
        sys.exit(1)

    with open(SMOKE_FILE, "r", encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f][:max_prompts]

    console.print(f"Loaded [bold]{len(prompts)}[/bold] smoke prompts from {SMOKE_FILE.name}")
    hw_info = get_hardware_info()
    thermal = get_thermal_state()
    console.print(f"Hardware: [dim]{hw_info}[/dim] | Thermal State: [dim]{thermal}[/dim]\n")

    store = TraceStore(db_path=TRACES_DIR / "camir_traces.sqlite")

    # Initialize Tiers
    hosted_tier = HostedAPITier(
        model_id="qwen/qwen-2.5-72b-instruct",
        tier_name="large",
    )
    console.print(f"Large Tier: [green]{hosted_tier.model_id}[/green] ({'Mock Mode' if hosted_tier.is_mock else 'Live Hosted API'})")

    if not mock_local:
        small_tier = LocalMLXTier(
            model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            tier_name="small",
        )
        console.print(f"Small Tier: [green]{small_tier.model_id}[/green] (MLX-LM Local)")
    else:
        # Fast mock local tier for instant testing
        small_tier = None
        console.print("Small Tier: [yellow]Fast Mock Mode (--mock-local flag enabled)[/yellow]")

    results_table = Table(title="Smoke Pipeline Execution Summary")
    results_table.add_column("Prompt ID", style="cyan")
    results_table.add_column("Dataset", style="blue")
    results_table.add_column("Tier", style="magenta")
    results_table.add_column("Tokens (In/Out)", style="white")
    results_table.add_column("TTFT (ms)", style="yellow")
    results_table.add_column("Total Lat (ms)", style="yellow")
    results_table.add_column("Energy (J)", style="green")
    results_table.add_column("Cost ($)", style="green")
    results_table.add_column("Status", style="bold green")

    validated_count = 0
    total_executions = 0

    for item in track(prompts, description="Processing smoke prompts..."):
        prompt_id = item["prompt_id"]
        dataset = item["dataset"]
        prompt_text = item["prompt"]

        # 1. Run through Small Tier
        total_executions += 1
        config_small = {
            "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "temperature": 0.0,
            "max_tokens": 64,
            "seed": 42,
            "quantization": "4bit",
        }
        hash_small = compute_config_hash(config_small)

        if small_tier is not None:
            res_s = small_tier.generate(prompt=prompt_text, max_tokens=64, temperature=0.0, seed=42)
        else:
            # Fast mock small tier
            res_s = hosted_tier._generate_mock(prompt_text, max_tokens=64)
            res_s.model_id = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
            res_s.energy_j = 0.42

        # Extract logprob stats
        mean_lp = sum(lp.logprob for lp in res_s.logprobs) / len(res_s.logprobs) if res_s.logprobs else None
        min_lp = min((lp.logprob for lp in res_s.logprobs), default=None)
        margins = [lp.top2_margin for lp in res_s.logprobs if lp.top2_margin is not None]
        mean_margin = (sum(margins) / len(margins)) if margins else None

        row_small = TraceRow(
            trace_id=str(uuid.uuid4()),
            prompt_id=prompt_id,
            dataset=dataset,
            tier="small",
            model_id=res_s.model_id,
            config_hash=hash_small,
            prompt_text=prompt_text,
            output_text=res_s.text,
            prompt_tokens=res_s.prompt_tokens,
            output_tokens=res_s.output_tokens,
            mean_logprob=round(mean_lp, 4) if mean_lp is not None else None,
            min_logprob=round(min_lp, 4) if min_lp is not None else None,
            mean_top2_margin=round(mean_margin, 4) if mean_margin is not None else None,
            cost_usd=0.0,  # Local model = $0 token cost
            t_prefill_s=round(res_s.timing.t_prefill_s, 4),
            t_ttft_s=round(res_s.timing.t_ttft_s, 4),
            t_decode_s=round(res_s.timing.t_decode_s, 4),
            t_total_s=round(res_s.timing.t_total_s, 4),
            tpot_s=round(res_s.timing.tpot_s, 4),
            energy_j=round(res_s.energy_j, 4) if res_s.energy_j else None,
            thermal_state=get_thermal_state(),
            hardware_info=hw_info,
        )
        store.save_trace(row_small, overwrite=True)
        validated_count += 1
        results_table.add_row(
            prompt_id[:16],
            dataset,
            "small",
            f"{row_small.prompt_tokens}/{row_small.output_tokens}",
            f"{row_small.t_ttft_s * 1000:.1f}",
            f"{row_small.t_total_s * 1000:.1f}",
            f"{row_small.energy_j:.2f} J" if row_small.energy_j else "N/A",
            "$0.0000",
            "VALID",
        )

        # 2. Run through Large Tier (Hosted API)
        total_executions += 1
        config_large = {
            "model_id": hosted_tier.model_id,
            "temperature": 0.0,
            "max_tokens": 64,
            "seed": 42,
            "quantization": "fp16",
        }
        hash_large = compute_config_hash(config_large)
        res_l = hosted_tier.generate(prompt=prompt_text, max_tokens=64, temperature=0.0, seed=42)
        cost_l = calculate_token_cost(hosted_tier.model_id, res_l.prompt_tokens, res_l.output_tokens)

        row_large = TraceRow(
            trace_id=str(uuid.uuid4()),
            prompt_id=prompt_id,
            dataset=dataset,
            tier="large",
            model_id=res_l.model_id,
            config_hash=hash_large,
            prompt_text=prompt_text,
            output_text=res_l.text,
            prompt_tokens=res_l.prompt_tokens,
            output_tokens=res_l.output_tokens,
            cost_usd=cost_l,
            t_prefill_s=round(res_l.timing.t_prefill_s, 4),
            t_ttft_s=round(res_l.timing.t_ttft_s, 4),
            t_decode_s=round(res_l.timing.t_decode_s, 4),
            t_total_s=round(res_l.timing.t_total_s, 4),
            tpot_s=round(res_l.timing.tpot_s, 4),
            energy_j=0.0,
            thermal_state=get_thermal_state(),
            hardware_info=hw_info,
        )
        store.save_trace(row_large, overwrite=True)
        validated_count += 1
        results_table.add_row(
            prompt_id[:16],
            dataset,
            "large",
            f"{row_large.prompt_tokens}/{row_large.output_tokens}",
            f"{row_large.t_ttft_s * 1000:.1f}",
            f"{row_large.t_total_s * 1000:.1f}",
            "0.00 J",
            f"${cost_l:.6f}",
            "VALID",
        )

    console.print(results_table)

    # Export to Parquet
    parquet_path = TRACES_DIR / "smoke_traces.parquet"
    exported_rows = store.export_to_parquet(parquet_path)

    console.print(f"\n[bold green]✓ Phase 1 Exit Gate Satisfied![/bold green]")
    console.print(f"Total Prompts Processed: [bold]{len(prompts)}[/bold]")
    console.print(f"Total Trace Rows Validated: [bold]{validated_count}/{total_executions}[/bold]")
    console.print(f"SQLite Store: [cyan]{store.db_path}[/cyan] ({store.count_traces()} total rows)")
    console.print(f"Parquet Export: [cyan]{parquet_path}[/cyan] ({exported_rows} rows)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CAMIR Phase 1 smoke test suite.")
    parser.add_argument("--mock-local", action="store_true", help="Run local tier in mock mode for fast testing.")
    parser.add_argument("--max-prompts", type=int, default=20, help="Max prompts to run (default 20).")
    args = parser.parse_args()

    run_smoke_pipeline(mock_local=args.mock_local, max_prompts=args.max_prompts)
