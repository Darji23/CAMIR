"""Unit tests for Tier interfaces and generation contracts."""

import pytest
from camir.tiers.hosted_api import HostedAPITier
from camir.traces.schema import TierResult


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
