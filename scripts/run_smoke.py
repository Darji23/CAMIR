"""End-to-End Phase 1 Smoke Pipeline Driver.

Runs the 20-prompt smoke dataset through:
1. Small Tier (MLX-LM Qwen2.5-1.5B 4-bit)
2. Medium Tier (MLX-LM Qwen2.5-7B 4-bit, fallback verifier for P5)
3. Large Tier (Hosted API Qwen2.5-72B / Mock)

Saves validated TraceRows to SQLite, exports to Parquet,
and automatically stores execution logs and structured summaries in checkpoints/phase_1/.

Execution:
    python scripts/run_smoke.py
    python scripts/run_smoke.py --mock-local   # fast dry-run for small/medium without weights
    python scripts/run_smoke.py --skip-medium  # run small and large tiers only
"""

import argparse
import json
import os
from pathlib import Path
import sys
import uuid
from typing import List, Optional
from rich.console import Console
from rich.progress import track
from rich.table import Table

from camir.accounting.cost import calculate_token_cost
from camir.tiers.hosted_api import HostedAPITier
from camir.tiers.mlx_local import LocalMLXTier, MediumMLXTier
from camir.traces.schema import TokenLogprob, TraceRow
from camir.traces.store import TraceStore
from camir.utils.checkpoint import CheckpointManager, TeeCapture
from camir.utils.hardware import get_hardware_info, get_thermal_state
from camir.utils.hashing import compute_config_hash

console = Console()
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SMOKE_FILE = DATA_DIR / "smoke_20.jsonl"
TRACES_DIR = Path(__file__).resolve().parent.parent / "traces"


def extract_mc_answer_logprob(logprobs: List[TokenLogprob]) -> Optional[float]:
    """Extract log-probability of the answer choice token (A, B, C, D) for multiple-choice tasks.

    Scans the first 20 output tokens to handle models that prefix the answer
    with a preamble (e.g. 'The answer is B' -> B is token 4, not token 0).
    Returns the logprob of the first A/B/C/D token found, or None if not found.
    """
    if not logprobs:
        return None
    for lp in logprobs[:20]:
        token_clean = lp.token_str.strip().strip("():.").upper()
        if token_clean in {"A", "B", "C", "D"}:
            return lp.logprob
    return None


def run_smoke_pipeline(
    skip_medium: bool = False,
    max_prompts: int = 20,
):
    ckpt_mgr = CheckpointManager(phase=1)
    console_log_path = ckpt_mgr.get_console_log_path("smoke")

    with TeeCapture(console_log_path):
        console.rule("[bold cyan]CAMIR Phase 1: 20-Prompt Smoke Pipeline (Small, Medium, Large)[/bold cyan]")

        if not SMOKE_FILE.exists():
            console.print(f"[bold red]Error: Smoke dataset not found at {SMOKE_FILE}.[/bold red]")
            console.print("Run: make data (or python camir/data/download_and_prep.py) first.")
            sys.exit(1)

        with open(SMOKE_FILE, "r", encoding="utf-8") as f:
            prompts = [json.loads(line) for line in f][:max_prompts]

        console.print(f"Loaded [bold]{len(prompts)}[/bold] smoke prompts from {SMOKE_FILE.name}")
        hw_info = get_hardware_info()
        thermal = get_thermal_state()
        console.print(f"Hardware: [dim]{hw_info}[/dim] | Thermal State: [dim]{thermal}[/dim]\n")

        store = TraceStore(db_path=TRACES_DIR / "camir_traces.sqlite")

        # Initialize Tiers (all real models, mock eliminated)
        hosted_tier = HostedAPITier(
            tier_name="large",
        )
        provider_name = "Fireworks AI" if ("fireworks" in (hosted_tier.base_url or "").lower() or "fireworks" in hosted_tier.model_id.lower()) else "OpenAI-compatible"
        console.print(f"Large Tier:  [green]{hosted_tier.model_id}[/green] (Live API via {provider_name})")

        small_tier = LocalMLXTier(
            model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            tier_name="small",
        )
        console.print(f"Small Tier:  [green]{small_tier.model_id}[/green] (MLX-LM Local)")
        if not skip_medium:
            medium_tier = MediumMLXTier(
                model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
                tier_name="medium",
            )
            console.print(f"Medium Tier: [green]{medium_tier.model_id}[/green] (MLX-LM Local, P5 verifier fallback)")
        else:
            medium_tier = None
            console.print("Medium Tier: [yellow]Skipped via --skip-medium[/yellow]")

        results_table = Table(title="Smoke Pipeline Multi-Tier Execution Summary")
        results_table.add_column("Prompt ID", style="cyan")
        results_table.add_column("Dataset", style="blue")
        results_table.add_column("Tier", style="magenta")
        results_table.add_column("Tokens (In/Out)", style="white")
        results_table.add_column("Ans Logprob", style="cyan")
        results_table.add_column("TTFT (ms)", style="yellow")
        results_table.add_column("Total Lat (ms)", style="yellow")
        results_table.add_column("Energy (J, 12.5W est)", style="green")
        results_table.add_column("Cost ($)", style="green")
        results_table.add_column("Status", style="bold green")

        validated_count = 0
        total_executions = 0

        for item in track(prompts, description="Processing smoke prompts across tiers..."):
            prompt_id = item["prompt_id"]
            dataset = item["dataset"]
            prompt_text = item["prompt"]
            is_mc = dataset in {"arc_easy", "mmlu"}

            # -------------------------------------------------------------
            # 1. Small Tier (Qwen2.5-1.5B)
            # -------------------------------------------------------------
            total_executions += 1
            config_small = {
                "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "temperature": 0.0,
                "max_tokens": 64,
                "seed": 42,
                "quantization": "4bit",
                "prompt_template": "chat_instruct",
                "hardware_profile": "apple_silicon",
            }
            hash_small = compute_config_hash(config_small)

            res_s = small_tier.generate(prompt=prompt_text, max_tokens=128, temperature=0.0, seed=42)

            mean_lp_s = sum(lp.logprob for lp in res_s.logprobs) / len(res_s.logprobs) if res_s.logprobs else None
            min_lp_s = min((lp.logprob for lp in res_s.logprobs), default=None)
            margins_s = [lp.top2_margin for lp in res_s.logprobs if lp.top2_margin is not None]
            mean_margin_s = (sum(margins_s) / len(margins_s)) if margins_s else None
            ans_lp_s = extract_mc_answer_logprob(res_s.logprobs) if is_mc else None

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
                mean_logprob=round(mean_lp_s, 4) if mean_lp_s is not None else None,
                min_logprob=round(min_lp_s, 4) if min_lp_s is not None else None,
                mean_top2_margin=round(mean_margin_s, 4) if mean_margin_s is not None else None,
                answer_token_logprob=round(ans_lp_s, 4) if ans_lp_s is not None else None,
                cost_usd=0.0,
                t_prefill_s=round(res_s.timing.t_prefill_s, 4),
                t_ttft_s=round(res_s.timing.t_ttft_s, 4),
                t_decode_s=round(res_s.timing.t_decode_s, 4),
                t_total_s=round(res_s.timing.t_total_s, 4),
                tpot_s=round(res_s.timing.tpot_s, 4),
                energy_j=round(res_s.energy_j, 4) if res_s.energy_j else None,
                energy_method=res_s.energy_method,
                cache_disabled=True,
                thermal_state=get_thermal_state(),
                hardware_profile=res_s.hardware_profile,
                hardware_info=hw_info,
            )

            store.save_trace(row_small, overwrite=True)
            validated_count += 1
            cost_s = calculate_token_cost(res_s.model_id, res_s.prompt_tokens, res_s.output_tokens)
            results_table.add_row(
                prompt_id[:16],
                dataset,
                "small",
                f"{row_small.prompt_tokens}/{row_small.output_tokens}",
                f"{row_small.answer_token_logprob:.2f}" if row_small.answer_token_logprob is not None else "-",
                f"{row_small.t_ttft_s * 1000:.1f}",
                f"{row_small.t_total_s * 1000:.1f}",
                f"{row_small.energy_j:.2f} J" if row_small.energy_j else "N/A",
                f"local (${cost_s:.2f})",
                "VALID",
            )

            # -------------------------------------------------------------
            # 2. Medium Tier (Qwen2.5-7B) if not skipped
            # -------------------------------------------------------------
            if not skip_medium:
                total_executions += 1
                config_med = {
                    "model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
                    "temperature": 0.0,
                    "max_tokens": 64,
                    "seed": 42,
                    "quantization": "4bit",
                    "prompt_template": "chat_instruct",
                    "hardware_profile": "apple_silicon",
                }
                hash_med = compute_config_hash(config_med)

                res_m = medium_tier.generate(prompt=prompt_text, max_tokens=128, temperature=0.0, seed=42)

                mean_lp_m = sum(lp.logprob for lp in res_m.logprobs) / len(res_m.logprobs) if res_m.logprobs else None
                min_lp_m = min((lp.logprob for lp in res_m.logprobs), default=None)
                margins_m = [lp.top2_margin for lp in res_m.logprobs if lp.top2_margin is not None]
                mean_margin_m = (sum(margins_m) / len(margins_m)) if margins_m else None
                ans_lp_m = extract_mc_answer_logprob(res_m.logprobs) if is_mc else None

                row_med = TraceRow(
                    trace_id=str(uuid.uuid4()),
                    prompt_id=prompt_id,
                    dataset=dataset,
                    tier="medium",
                    model_id=res_m.model_id,
                    config_hash=hash_med,
                    prompt_text=prompt_text,
                    output_text=res_m.text,
                    prompt_tokens=res_m.prompt_tokens,
                    output_tokens=res_m.output_tokens,
                    mean_logprob=round(mean_lp_m, 4) if mean_lp_m is not None else None,
                    min_logprob=round(min_lp_m, 4) if min_lp_m is not None else None,
                    mean_top2_margin=round(mean_margin_m, 4) if mean_margin_m is not None else None,
                    answer_token_logprob=round(ans_lp_m, 4) if ans_lp_m is not None else None,
                    cost_usd=0.0,
                    t_prefill_s=round(res_m.timing.t_prefill_s, 4),
                    t_ttft_s=round(res_m.timing.t_ttft_s, 4),
                    t_decode_s=round(res_m.timing.t_decode_s, 4),
                    t_total_s=round(res_m.timing.t_total_s, 4),
                    tpot_s=round(res_m.timing.tpot_s, 4),
                    energy_j=round(res_m.energy_j, 4) if res_m.energy_j else None,
                    energy_method=res_m.energy_method,
                    cache_disabled=True,
                    thermal_state=get_thermal_state(),
                    hardware_profile=res_m.hardware_profile,
                    hardware_info=hw_info,
                )

                store.save_trace(row_med, overwrite=True)
                validated_count += 1
                cost_m = calculate_token_cost(res_m.model_id, res_m.prompt_tokens, res_m.output_tokens)
                results_table.add_row(
                    prompt_id[:16],
                    dataset,
                    "medium",
                    f"{row_med.prompt_tokens}/{row_med.output_tokens}",
                    f"{row_med.answer_token_logprob:.2f}" if row_med.answer_token_logprob is not None else "-",
                    f"{row_med.t_ttft_s * 1000:.1f}",
                    f"{row_med.t_total_s * 1000:.1f}",
                    f"{row_med.energy_j:.2f} J" if row_med.energy_j else "N/A",
                    f"local (${cost_m:.2f})",
                    "VALID",
                )

            # -------------------------------------------------------------
            # 3. Large Tier (Hosted API / Mock)
            # -------------------------------------------------------------
            total_executions += 1
            config_large = {
                "model_id": hosted_tier.model_id,
                "temperature": 0.0,
                "max_tokens": 64,
                "seed": 42,
                "quantization": "fp16",
                "prompt_template": "chat_instruct",
                "hardware_profile": "cloud",
            }
            hash_large = compute_config_hash(config_large)
            # For MC tasks, instruct the model to answer with only the letter
            # so the A/B/C/D token appears early in the output (aids logprob extraction)
            mc_system_prompt = (
                "You are a concise answer assistant. For multiple-choice questions, "
                "respond with ONLY the single letter of the correct answer (A, B, C, or D). "
                "Do not add any explanation."
            ) if is_mc else None
            res_l = hosted_tier.generate(
                prompt=prompt_text,
                max_tokens=64,
                temperature=0.0,
                seed=42,
                system_prompt=mc_system_prompt,
            )
            cost_l = calculate_token_cost(hosted_tier.model_id, res_l.prompt_tokens, res_l.output_tokens)
            ans_lp_l = extract_mc_answer_logprob(res_l.logprobs) if is_mc else None

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
                answer_token_logprob=round(ans_lp_l, 4) if ans_lp_l is not None else None,
                cost_usd=cost_l,
                t_prefill_s=round(res_l.timing.t_prefill_s, 4),
                t_ttft_s=round(res_l.timing.t_ttft_s, 4),
                t_decode_s=round(res_l.timing.t_decode_s, 4),
                t_total_s=round(res_l.timing.t_total_s, 4),
                tpot_s=round(res_l.timing.tpot_s, 4),
                energy_j=0.0,
                energy_method=res_l.energy_method,
                cache_disabled=True,
                thermal_state=get_thermal_state(),
                hardware_profile=res_l.hardware_profile,
                hardware_info=hw_info,
            )

            store.save_trace(row_large, overwrite=True)
            validated_count += 1
            cost_label_l = f"${cost_l:.8f}" if cost_l > 0 else "$0.00000000"
            results_table.add_row(
                prompt_id[:16],
                dataset,
                "large",
                f"{row_large.prompt_tokens}/{row_large.output_tokens}",
                f"{row_large.answer_token_logprob:.2f}" if row_large.answer_token_logprob is not None else "-",
                f"{row_large.t_ttft_s * 1000:.1f}",
                f"{row_large.t_total_s * 1000:.1f}",
                "0.00 J",
                cost_label_l,
                "VALID",
            )

        console.print(results_table)

        # Export to Parquet
        parquet_path = TRACES_DIR / "smoke_traces.parquet"
        exported_rows = store.export_to_parquet(parquet_path)

        summary_text = (
            f"Phase 1 Smoke Pipeline Execution Summary:\n"
            f"- Prompts Processed: {len(prompts)}\n"
            f"- Tiers Run: Small (Qwen-1.5B), {'Medium (Qwen-7B), ' if not skip_medium else ''}Large (Qwen-72B)\n"
            f"- Trace Rows Validated: {validated_count}/{total_executions}\n"
            f"- SQLite Database: {store.db_path} ({store.count_traces()} total rows)\n"
            f"- Parquet Export: {parquet_path} ({exported_rows} rows)\n"
            f"- Energy Accounting: Labeled estimated (12.5 W active power) across all local rows\n"
            f"- Cache Status: Disabled across all inference calls\n"
            f"- MC Answer Token Logprob: Recorded separately for ARC-Easy and MMLU\n"
        )

        ckpt_mgr.save_output_checkpoint(
            output_name="smoke_summary",
            content=summary_text,
            metadata={
                "skip_medium": skip_medium,
                "max_prompts": max_prompts,
                "validated_rows": validated_count,
            },
        )

        console.print(f"\n[bold green]✓ Phase 1 Exit Gate Satisfied![/bold green]")
        console.print(f"Total Prompts Processed: [bold]{len(prompts)}[/bold]")
        console.print(f"Total Trace Rows Validated: [bold]{validated_count}/{total_executions}[/bold]")
        console.print(f"SQLite Store: [cyan]{store.db_path}[/cyan] ({store.count_traces()} total rows)")
        console.print(f"Parquet Export: [cyan]{parquet_path}[/cyan] ({exported_rows} rows)")
        console.print(f"Checkpoint Saved: [cyan]{ckpt_mgr.get_output_path('smoke_summary')}[/cyan]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CAMIR Phase 1 smoke test suite.")
    parser.add_argument("--skip-medium", action="store_true", help="Skip medium tier.")
    parser.add_argument("--max-prompts", type=int, default=20, help="Max prompts to run (default 20).")
    args = parser.parse_args()

    run_smoke_pipeline(
        skip_medium=args.skip_medium,
        max_prompts=args.max_prompts,
    )
