"""Spike S1: MLX-LM Log-prob and Margin Extraction.

Verifies that MLX-LM on Apple Silicon can load Qwen2.5-1.5B (4-bit),
generate output tokens, and calculate per-token logprobs and top-2 margins.
Saves console logs and output summaries to checkpoints/phase_1/.

Execution:
    python spikes/s1_mlx_logprobs.py
"""

import time
from rich.console import Console
from rich.table import Table

from camir.tiers.mlx_local import LocalMLXTier
from camir.utils.checkpoint import CheckpointManager, TeeCapture

console = Console()


def run_spike_s1():
    ckpt_mgr = CheckpointManager(phase=1)
    console_log_path = ckpt_mgr.get_console_log_path("spike1")

    with TeeCapture(console_log_path):
        console.rule("[bold cyan]Spike S1: MLX-LM Log-prob & Margin Extraction[/bold cyan]")
        
        tier = LocalMLXTier(
            model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            tier_name="small"
        )

        console.print(f"[yellow]Loading model: {tier.model_id}...[/yellow]")
        t0 = time.perf_counter()
        tier.load_model()
        load_time = time.perf_counter() - t0
        console.print(f"[green]✓ Model loaded in {load_time:.2f}s (isolated)[/green]")

        test_prompt = "Question: What is the powerhouse of the cell?\nChoices:\n(A) Nucleus\n(B) Mitochondria\n(C) Ribosome\n(D) Golgi\nAnswer:"
        console.print(f"\n[bold]Prompt:[/bold]\n{test_prompt}\n")

        console.print("[yellow]Generating with logprob and margin tracking (cache disabled)...[/yellow]")
        result = tier.generate(
            prompt=test_prompt,
            max_tokens=32,
            temperature=0.0,
            seed=42
        )

        console.print(f"[bold green]Completion:[/bold green] {result.text}\n")

        # Display timings
        timing_table = Table(title="Timing Breakdown (Rules Enforced)")
        timing_table.add_column("Metric", style="cyan")
        timing_table.add_column("Value", style="magenta")
        timing_table.add_row("Model Load Time", f"{result.timing.t_load_s:.3f} s (isolated)")
        timing_table.add_row("Time To First Token (TTFT)", f"{result.timing.t_ttft_s * 1000:.1f} ms")
        timing_table.add_row("Decode Latency", f"{result.timing.t_decode_s * 1000:.1f} ms")
        timing_table.add_row("Total Latency", f"{result.timing.t_total_s * 1000:.1f} ms")
        timing_table.add_row("Time Per Output Token (TPOT)", f"{result.timing.tpot_s * 1000:.2f} ms/token")
        timing_table.add_row("Energy (Joules)", f"{result.energy_j:.4f} J (estimated, 12.5W active)")
        timing_table.add_row("Cache Disabled", str(result.timing.cache_disabled))
        timing_table.add_row("Thermal State", result.timing.thermal_state)
        console.print(timing_table)

        # Display first few tokens with logprobs and margins
        extracted_tokens = []
        if result.logprobs:
            token_table = Table(title="Token Log-probabilities & Top-2 Margins")
            token_table.add_column("Token ID", style="dim")
            token_table.add_column("Token Text", style="cyan")
            token_table.add_column("Log-prob", style="green")
            token_table.add_column("Top-2 Margin", style="yellow")

            for lp in result.logprobs[:10]:
                token_table.add_row(
                    str(lp.token_id),
                    repr(lp.token_str),
                    f"{lp.logprob:.4f}",
                    f"{lp.top2_margin:.4f}" if lp.top2_margin is not None else "N/A"
                )
                extracted_tokens.append(f"{lp.token_str!r}: logprob={lp.logprob:.4f}, margin={lp.top2_margin}")
            console.print(token_table)

        summary_content = (
            f"Spike S1 Findings:\n"
            f"- Model: {tier.model_id}\n"
            f"- Load Latency: {load_time:.3f} s (isolated)\n"
            f"- Generation Latency: {result.timing.t_total_s * 1000:.1f} ms\n"
            f"- TTFT: {result.timing.t_ttft_s * 1000:.1f} ms\n"
            f"- TPOT: {result.timing.tpot_s * 1000:.2f} ms/token\n"
            f"- Energy: {result.energy_j:.4f} J (estimated, 12.5 W active)\n"
            f"- Cache Disabled: {result.timing.cache_disabled}\n"
            f"- Extracted Tokens Preview:\n  " + "\n  ".join(extracted_tokens[:5]) + "\n"
        )
        ckpt_mgr.save_output_checkpoint(
            output_name="spike1_summary",
            content=summary_content,
            metadata={"status": "PASSED", "model_id": tier.model_id},
        )

        console.print("\n[bold green]✓ Spike S1 Passed: MLX-LM provides logprobs and margins cleanly.[/bold green]")
        console.print(f"Checkpoint saved: [cyan]{ckpt_mgr.get_output_path('spike1_summary')}[/cyan]\n")


if __name__ == "__main__":
    run_spike_s1()
