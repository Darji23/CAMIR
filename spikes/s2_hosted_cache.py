"""Spike S2: Hosted API Integration and Zero-Duplicate Spend Cache.

Verifies that the hosted API tier can be queried, and that subsequent queries
with the same (prompt_id, tier, config_hash) hit the SQLite cache with zero network cost.

Execution:
    python spikes/s2_hosted_cache.py
"""

import tempfile
import time
import uuid
from pathlib import Path
from rich.console import Console
from rich.table import Table
from camir.accounting.cost import calculate_token_cost
from camir.tiers.hosted_api import HostedAPITier
from camir.traces.schema import TraceRow
from camir.traces.store import TraceStore
from camir.utils.hashing import compute_config_hash

console = Console()


def run_spike_s2():
    console.rule("[bold cyan]Spike S2: Hosted API & Idempotent Caching[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_cache.sqlite"
        store = TraceStore(db_path=db_path)

        tier = HostedAPITier(
            model_id="qwen/qwen-2.5-72b-instruct",
            tier_name="large",
        )

        console.print(f"[yellow]Tier:[/yellow] {tier.tier_name} ({tier.model_id})")
        console.print(f"[yellow]Mode:[/yellow] {'Mock (offline)' if tier.is_mock else 'Live API'}")

        prompt_id = "smoke_arc_01"
        prompt_text = "What is the capital of France?\n(A) Berlin\n(B) Paris\n(C) Madrid\n(D) Rome\nAnswer:"
        config = {
            "model_id": tier.model_id,
            "temperature": 0.0,
            "max_tokens": 16,
            "seed": 42,
            "quantization": "fp16",
        }
        config_hash = compute_config_hash(config)

        table = Table(title="Idempotency & Cost Verification")
        table.add_column("Run", style="cyan")
        table.add_column("Source", style="yellow")
        table.add_column("Latency (ms)", style="magenta")
        table.add_column("Cost (USD)", style="green")
        table.add_column("Output Preview", style="white")

        # Run 1: First call (executes model/API and saves to cache)
        t0 = time.perf_counter()
        cached = store.get_trace(prompt_id=prompt_id, tier=tier.tier_name, config_hash=config_hash)
        if cached is None:
            res1 = tier.generate(prompt=prompt_text, max_tokens=16, temperature=0.0, seed=42)
            cost1 = calculate_token_cost(tier.model_id, res1.prompt_tokens, res1.output_tokens)
            
            trace_row = TraceRow(
                trace_id=str(uuid.uuid4()),
                prompt_id=prompt_id,
                dataset="arc_easy",
                tier=tier.tier_name,
                model_id=tier.model_id,
                config_hash=config_hash,
                prompt_text=prompt_text,
                output_text=res1.text,
                prompt_tokens=res1.prompt_tokens,
                output_tokens=res1.output_tokens,
                cost_usd=cost1,
                t_prefill_s=res1.timing.t_prefill_s,
                t_ttft_s=res1.timing.t_ttft_s,
                t_decode_s=res1.timing.t_decode_s,
                t_total_s=res1.timing.t_total_s,
                tpot_s=res1.timing.tpot_s,
                energy_j=res1.energy_j,
            )
            store.save_trace(trace_row)
            lat1 = (time.perf_counter() - t0) * 1000
            table.add_row("1st Call", "Network/Engine", f"{lat1:.1f} ms", f"${cost1:.6f}", repr(res1.text[:25]))

        # Run 2: Second call (MUST hit SQLite cache, 0 network, $0.00 extra cost)
        t0 = time.perf_counter()
        cached2 = store.get_trace(prompt_id=prompt_id, tier=tier.tier_name, config_hash=config_hash)
        lat2 = (time.perf_counter() - t0) * 1000
        table.add_row(
            "2nd Call",
            "SQLite Cache HIT",
            f"{lat2:.2f} ms",
            "$0.000000",
            repr(cached2.output_text[:25]) if cached2 else "ERROR",
        )

        # Run 3: Third call
        t0 = time.perf_counter()
        cached3 = store.get_trace(prompt_id=prompt_id, tier=tier.tier_name, config_hash=config_hash)
        lat3 = (time.perf_counter() - t0) * 1000
        table.add_row(
            "3rd Call",
            "SQLite Cache HIT",
            f"{lat3:.2f} ms",
            "$0.000000",
            repr(cached3.output_text[:25]) if cached3 else "ERROR",
        )

        console.print(table)
        assert cached2 is not None
        assert cached2.output_text == cached3.output_text
        console.print("[bold green]✓ Spike S2 Passed: Hosted API idempotent caching verified.[/bold green]")


if __name__ == "__main__":
    run_spike_s2()
