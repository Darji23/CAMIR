"""Unit tests for trace schema and trace store."""

import tempfile
from pathlib import Path
import pytest
from pydantic import ValidationError
from camir.traces.schema import TimingBreakdown, TokenLogprob, TraceRow
from camir.traces.store import TraceStore


def test_trace_row_validation():
    # Valid row
    row = TraceRow(
        trace_id="test-uuid-1",
        prompt_id="arc_001",
        dataset="arc_easy",
        tier="small",
        model_id="Qwen/Qwen2.5-1.5B-Instruct-4bit",
        config_hash="abc123hash",
        prompt_text="What is 2+2?",
        output_text="4",
        prompt_tokens=5,
        output_tokens=1,
        cost_usd=0.0,
        t_prefill_s=0.01,
        t_ttft_s=0.015,
        t_decode_s=0.005,
        t_total_s=0.02,
        tpot_s=0.005,
        energy_j=0.45,
    )
    assert row.tier == "small"
    assert row.cost_usd == 0.0

    # Invalid tier
    with pytest.raises(ValidationError):
        TraceRow(
            trace_id="test-uuid-2",
            prompt_id="arc_001",
            dataset="arc_easy",
            tier="giant",  # Invalid!
            model_id="test",
            config_hash="abc",
            prompt_text="test",
            output_text="test",
            prompt_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            t_prefill_s=0.01,
            t_ttft_s=0.01,
            t_decode_s=0.01,
            t_total_s=0.02,
            tpot_s=0.01,
        )


def test_trace_store_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_traces.sqlite"
        store = TraceStore(db_path=db_path)

        row1 = TraceRow(
            trace_id="trace-001",
            prompt_id="prompt-A",
            dataset="mmlu",
            tier="small",
            model_id="Qwen/Qwen2.5-1.5B-Instruct-4bit",
            config_hash="hash-1",
            prompt_text="What is mitosis?",
            output_text="Cell division",
            prompt_tokens=10,
            output_tokens=5,
            cost_usd=0.0,
            t_prefill_s=0.02,
            t_ttft_s=0.025,
            t_decode_s=0.05,
            t_total_s=0.07,
            tpot_s=0.0125,
            energy_j=1.2,
        )

        # Insert row1
        assert store.save_trace(row1) is True
        assert store.count_traces() == 1

        # Duplicate insert without overwrite should be ignored
        assert store.save_trace(row1, overwrite=False) is False
        assert store.count_traces() == 1

        # Lookup
        cached = store.get_trace(prompt_id="prompt-A", tier="small", config_hash="hash-1")
        assert cached is not None
        assert cached.trace_id == "trace-001"
        assert cached.output_text == "Cell division"

        # Non-existent lookup
        assert store.get_trace(prompt_id="prompt-B", tier="small", config_hash="hash-1") is None

        # Parquet export
        parquet_path = Path(tmpdir) / "test_export.parquet"
        exported_count = store.export_to_parquet(parquet_path)
        assert exported_count == 1
        assert parquet_path.exists()
