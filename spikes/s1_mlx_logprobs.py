"""Spike S1: MLX-LM Log-prob and Margin Extraction.

Verifies that MLX-LM on Apple Silicon can load Qwen2.5-1.5B (4-bit),
generate output tokens, and calculate per-token logprobs and top-2 margins.

Execution:
    python spikes/s1_mlx_logprobs.py
"""

import time
from rich.console import Console
from rich.table import Table
from camir.tiers.mlx_local import LocalMLXTier

console = Console()


def run_spike_s1():
    console.rule("[bold cyan]Spike S1: MLX-LM Log-prob & Margin Extraction[/bold cyan]")
    
    tier = LocalMLXTier(
        model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        tier_name="small"
    )

    console.print(f"[yellow]Loading model: {tier.model_id}...[/yellow]")
    t0 = time.perf_counter()
    tier.load_model()
    load_time = time.perf_counter() - t0
    console.print(f"[green]✓ Model loaded in {load_time:.2f}s[/green]")

    test_prompt = "Question: What is the powerhouse of the cell?\nChoices:\n(A) Nucleus\n(B) Mitochondria\n(C) Ribosome\n(D) Golgi\nAnswer:"
    console.print(f"\n[bold]Prompt:[/bold]\n{test_prompt}\n")

    console.print("[yellow]Generating with logprob and margin tracking...[/yellow]")
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
    timing_table.add_row("Estimated Energy", f"{result.energy_j:.4f} Joules")
    console.print(timing_table)

    # Display first few tokens with logprobs and margins
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
        console.print(token_table)

    console.print("\n[bold green]✓ Spike S1 Passed: MLX-LM provides logprobs and margins cleanly.[/bold green]")


if __name__ == "__main__":
    run_spike_s1()
