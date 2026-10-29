"""Unit tests for trace schema, hashing, and trace store."""

from pathlib import Path
import tempfile
import pytest
from pydantic import ValidationError

from camir.traces.schema import TimingBreakdown, TokenLogprob, TraceRow
from camir.traces.store import TraceStore
from camir.utils.hashing import compute_config_hash, get_library_versions


def test_trace_row_validation():
    # Valid row with answer_token_logprob, energy_method, and cache_disabled
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
        answer_token_logprob=-0.1234,
        cost_usd=0.0,
        t_prefill_s=0.01,
        t_ttft_s=0.015,
        t_decode_s=0.005,
        t_total_s=0.02,
        tpot_s=0.005,
        energy_j=0.45,
        energy_method="estimated",
        cache_disabled=True,
    )
    assert row.tier == "small"
    assert row.cost_usd == 0.0
    assert row.answer_token_logprob == -0.1234
    assert row.energy_method == "estimated"
    assert row.cache_disabled is True

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


def test_config_hash_covers_all_fields():
    """Verify config_hash covers model, quantization, max tokens, temperature, prompt template, and library versions."""
    base_config = {
        "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "quantization": "4bit",
        "max_tokens": 64,
        "temperature": 0.0,
        "seed": 42,
        "prompt_template": "chat_template_v1",
        "library_versions": {"mlx": "0.22.0", "python": "3.11.0"},
    }
    base_hash = compute_config_hash(base_config)
    assert len(base_hash) == 16

    # 1. Altering model must change hash
    c_model = dict(base_config, model_id="mlx-community/Qwen2.5-7B-Instruct-4bit")
    assert compute_config_hash(c_model) != base_hash

    # 2. Altering quantization must change hash
    c_quant = dict(base_config, quantization="8bit")
    assert compute_config_hash(c_quant) != base_hash

    # 3. Altering max_tokens must change hash
    c_tokens = dict(base_config, max_tokens=128)
    assert compute_config_hash(c_tokens) != base_hash

    # 4. Altering temperature must change hash
    c_temp = dict(base_config, temperature=0.7)
    assert compute_config_hash(c_temp) != base_hash

    # 5. Altering prompt_template must change hash
    c_tpl = dict(base_config, prompt_template="chat_template_v2")
    assert compute_config_hash(c_tpl) != base_hash

    # 6. Altering library_versions must change hash
    c_ver = dict(base_config, library_versions={"mlx": "0.23.0", "python": "3.11.0"})
    assert compute_config_hash(c_ver) != base_hash

    # 7. Altering hardware_profile must change hash
    c_hw = dict(base_config, hardware_profile="nvidia")
    assert compute_config_hash(c_hw) != base_hash



def test_trace_store_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_traces.sqlite"
        store = TraceStore(db_path=db_path)

        row1 = TraceRow(
            trace_id="trace-001",
            prompt_id="prompt-A",
            dataset="mmlu",
            tier="medium",
            model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
            config_hash="hash-1",
            prompt_text="What is mitosis?",
            output_text="Cell division",
            prompt_tokens=10,
            output_tokens=5,
            answer_token_logprob=-0.045,
            cost_usd=0.0,
            t_prefill_s=0.02,
            t_ttft_s=0.025,
            t_decode_s=0.05,
            t_total_s=0.07,
            tpot_s=0.0125,
            energy_j=1.2,
            energy_method="estimated",
            cache_disabled=True,
        )

        # Insert row1
        assert store.save_trace(row1) is True
        assert store.count_traces() == 1

        # Duplicate insert without overwrite should be ignored
        assert store.save_trace(row1, overwrite=False) is False
        assert store.count_traces() == 1

        # Lookup
        cached = store.get_trace(prompt_id="prompt-A", tier="medium", config_hash="hash-1")
        assert cached is not None
        assert cached.trace_id == "trace-001"
        assert cached.output_text == "Cell division"
        assert cached.answer_token_logprob == -0.045
        assert cached.energy_method == "estimated"
        assert cached.cache_disabled is True

        # Non-existent lookup
        assert store.get_trace(prompt_id="prompt-B", tier="medium", config_hash="hash-1") is None

        # Parquet export
        parquet_path = Path(tmpdir) / "test_export.parquet"
        exported_count = store.export_to_parquet(parquet_path)
        assert exported_count == 1
        assert parquet_path.exists()
