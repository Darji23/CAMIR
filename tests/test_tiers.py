"""Unit tests for Tier interfaces, Medium tier adapter, and checkpoint utilities."""

from pathlib import Path
import tempfile
import pytest

from camir.tiers.hosted_api import HostedAPITier
from camir.tiers.mlx_local import LocalMLXTier, MediumMLXTier
from camir.traces.schema import TierResult
from camir.utils.checkpoint import CheckpointManager, TeeCapture


def test_hosted_tier_mock_contract():
    tier = HostedAPITier(
        model_id="qwen/qwen-2.5-72b-instruct",
        tier_name="large",
        mock_mode=True,
    )
    assert tier.tier_name == "large"
    assert tier.model_id == "qwen/qwen-2.5-72b-instruct"
    assert tier.is_mock is True

    result = tier.generate("What is 10 + 10?", max_tokens=16)
    assert isinstance(result, TierResult)
    assert len(result.text) > 0
    assert result.prompt_tokens > 0
    assert result.output_tokens > 0
    assert result.timing.t_total_s > 0.0
    assert result.finish_reason == "stop"
    assert result.energy_method == "estimated"
    assert result.timing.cache_disabled is True


def test_medium_tier_adapter_initialization():
    """Verify MediumMLXTier defaults to Qwen2.5-7B-Instruct-4bit and tier 'medium'."""
    medium_tier = MediumMLXTier()
    assert medium_tier.tier_name == "medium"
    assert medium_tier.model_id == "mlx-community/Qwen2.5-7B-Instruct-4bit"

    small_tier = LocalMLXTier()
    assert small_tier.tier_name == "small"
    assert small_tier.model_id == "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def test_checkpoint_manager_lifecycle():
    """Verify CheckpointManager writes console logs and output summaries into dedicated txt files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(phase=1, base_dir=Path(tmpdir))

        # Save console log
        log_path = mgr.save_console_log("test_run", "Line 1: execution started\nLine 2: completed")
        assert log_path.exists()
        assert log_path.suffix == ".txt"
        assert "execution started" in log_path.read_text()

        # Save output checkpoint
        out_path = mgr.save_output_checkpoint(
            output_name="test_summary",
            content="Key findings:\n- Metric A = 0.95\n- Metric B = 12.3ms",
            metadata={"status": "PASSED", "eval_count": 20},
        )
        assert out_path.exists()
        assert out_path.suffix == ".txt"
        text = out_path.read_text()
        assert "CAMIR PHASE 1 CHECKPOINT" in text
        assert "Metric A = 0.95" in text
        assert "Hardware:" in text
