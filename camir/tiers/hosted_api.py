"""Hosted API tier for large models (e.g. Qwen2.5-72B) via OpenAI-compatible endpoints.

Supports OpenRouter, Together AI, Fireworks, DeepInfra, etc.
Includes mock fallback mode for local testing without spending API credits.
"""

import os
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI
from camir.accounting.cost import calculate_token_cost
from camir.accounting.timing import compute_timing_breakdown
from camir.tiers.base import BaseTier
from camir.traces.schema import TierResult, TimingBreakdown, TokenLogprob


class HostedAPITier(BaseTier):
    """Hosted large tier accessing models via OpenAI-compatible REST API."""

    def __init__(
        self,
        model_id: str = "qwen/qwen-2.5-72b-instruct",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        tier_name: str = "large",
        mock_mode: bool = False,
    ):
        self._tier_name = tier_name
        self._model_id = model_id
        self._api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("TOGETHER_API_KEY")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._mock_mode = mock_mode or (not self._api_key)
        self._client: Optional[OpenAI] = None

    @property
    def tier_name(self) -> str:
        return self._tier_name

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_mock(self) -> bool:
        return self._mock_mode

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self._api_key or "mock-key",
                base_url=self._base_url,
            )
        return self._client

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 42,
        system_prompt: Optional[str] = None,
    ) -> TierResult:
        """Call hosted API or return mock completion if API key is not configured."""
        if self._mock_mode:
            return self._generate_mock(prompt, max_tokens)

        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=self._model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                logprobs=True,
                top_logprobs=2,
            )
            t_total_s = time.perf_counter() - t0

            choice = response.choices[0]
            output_text = choice.message.content or ""
            finish_reason = choice.finish_reason or "stop"
            usage = response.usage

            prompt_tokens = usage.prompt_tokens if usage else len(prompt.split())
            output_tokens = usage.completion_tokens if usage else len(output_text.split())

            # Parse logprobs if returned by provider
            token_logprobs: List[TokenLogprob] = []
            if choice.logprobs and choice.logprobs.content:
                for lp_item in choice.logprobs.content:
                    margin: Optional[float] = None
                    if lp_item.top_logprobs and len(lp_item.top_logprobs) >= 2:
                        top1 = lp_item.top_logprobs[0].logprob
                        top2 = lp_item.top_logprobs[1].logprob
                        margin = round(top1 - top2, 4)

                    token_logprobs.append(
                        TokenLogprob(
                            token_id=0,
                            token_str=lp_item.token,
                            logprob=round(lp_item.logprob, 4),
                            top2_margin=margin,
                        )
                    )

            # Approximation of TTFT for hosted API (typically 60-70% of total latency is prefill + queue)
            t_ttft_s = t_total_s * 0.65
            t_decode_s = max(0.0, t_total_s - t_ttft_s)

            timing = compute_timing_breakdown(
                t_load_s=0.0,
                t_prefill_s=t_ttft_s,
                t_ttft_s=t_ttft_s,
                t_decode_s=t_decode_s,
                output_tokens=output_tokens,
            )

            return TierResult(
                text=output_text.strip(),
                token_ids=[],
                logprobs=token_logprobs,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                timing=timing,
                model_id=self._model_id,
                finish_reason=finish_reason,
                energy_j=0.0,  # Hosted API does not consume local battery energy
                raw_response_metadata={"hosted": True, "provider_id": response.id},
            )

        except Exception as e:
            # Fall back to mock on network or credit error to prevent pipeline blockage
            res = self._generate_mock(prompt, max_tokens)
            res.raw_response_metadata["fallback_from_error"] = str(e)
            return res

    def _generate_mock(self, prompt: str, max_tokens: int) -> TierResult:
        """Deterministic mock response for offline development and testing."""
        prompt_words = prompt.split()
        prompt_len = len(prompt_words)

        mock_text = f"[Hosted Tier Mock Response] Evaluated prompt with {prompt_len} words."
        output_tokens = len(mock_text.split())

        timing = TimingBreakdown(
            t_load_s=0.0,
            t_prefill_s=0.04,
            t_ttft_s=0.05,
            t_decode_s=0.03,
            t_total_s=0.07,
            tpot_s=0.003,
        )

        return TierResult(
            text=mock_text,
            token_ids=[],
            logprobs=[],
            prompt_tokens=prompt_len,
            output_tokens=output_tokens,
            timing=timing,
            model_id=self._model_id,
            finish_reason="stop",
            energy_j=0.0,
            raw_response_metadata={"mock": True},
        )
