"""Hosted API tier for large models (e.g. Qwen2.5-72B) via OpenAI-compatible endpoints.

Supports Fireworks AI (default recommended provider), OpenRouter, Together AI, etc.
Automatically loads API keys and endpoints from .env if present.
Includes mock fallback mode for local testing without spending API credits.
"""

import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI
from camir.accounting.cost import calculate_token_cost
from camir.accounting.timing import compute_timing_breakdown
from camir.tiers.base import BaseTier
from camir.traces.schema import TierResult, TimingBreakdown, TokenLogprob
from camir.utils.env import load_env

# Automatically load .env file from workspace root if not already loaded
load_env()

DEFAULT_FIREWORKS_MODEL = "accounts/fireworks/models/qwen3p8-max"
DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


class HostedAPITier(BaseTier):
    """Hosted large tier accessing models via OpenAI-compatible REST API.
    
    Seamlessly adapts to Fireworks AI when FIREWORKS_API_KEY is configured.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        tier_name: str = "large",
    ):
        load_env()
        self._tier_name = tier_name
        
        # 1. Resolve API Key (Fireworks -> OpenAI -> Together)
        self._api_key = (
            api_key
            or os.getenv("FIREWORKS_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("TOGETHER_API_KEY")
        )
        
        # 2. Resolve Base URL
        if base_url:
            self._base_url = base_url
        elif os.getenv("FIREWORKS_BASE_URL"):
            self._base_url = os.getenv("FIREWORKS_BASE_URL")
        elif os.getenv("FIREWORKS_API_KEY") and not os.getenv("OPENAI_BASE_URL"):
            # If FIREWORKS_API_KEY is present and no generic OPENAI_BASE_URL is specified, use Fireworks
            self._base_url = DEFAULT_FIREWORKS_BASE_URL
        else:
            self._base_url = os.getenv("OPENAI_BASE_URL")

        # 3. Resolve Model Identifier
        is_fireworks = bool(
            os.getenv("FIREWORKS_API_KEY")
            or (self._base_url and "fireworks" in self._base_url.lower())
        )
        
        if model_id:
            self._model_id = model_id
        elif os.getenv("HOSTED_MODEL_ID"):
            self._model_id = os.getenv("HOSTED_MODEL_ID")
        elif is_fireworks:
            self._model_id = DEFAULT_FIREWORKS_MODEL
        else:
            self._model_id = "accounts/fireworks/models/qwen3p8-max"

        self._client: Optional[OpenAI] = None

    @property
    def tier_name(self) -> str:
        return self._tier_name

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_mock(self) -> bool:
        return False

    @property
    def base_url(self) -> Optional[str]:
        return self._base_url

    @property
    def client(self) -> OpenAI:
        """Return initialized OpenAI client instance."""
        return self._get_client()

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "Hosted API key not configured. Mock mode is eliminated. "
                    "Please set FIREWORKS_API_KEY (or OPENAI_API_KEY) in .env"
                )
            self._client = OpenAI(
                api_key=self._api_key,
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
        """Call hosted API with real models; fails fast if call fails."""
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        try:
            # First attempt with logprobs and top_logprobs
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
            except Exception as param_err:
                # If provider rejects top_logprobs or logprobs parameter, retry without it
                if "logprobs" in str(param_err).lower() or "top_logprobs" in str(param_err).lower():
                    response = client.chat.completions.create(
                        model=self._model_id,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        seed=seed,
                    )
                else:
                    raise param_err

            t_total_s = time.perf_counter() - t0

            choice = response.choices[0]
            # Handle reasoning models where output may be in content or reasoning_content
            output_text = (choice.message.content or getattr(choice.message, "reasoning_content", "") or "").strip()
            finish_reason = choice.finish_reason or "stop"
            usage = response.usage

            prompt_tokens = usage.prompt_tokens if usage else len(prompt.split())
            output_tokens = usage.completion_tokens if usage else len(output_text.split())

            # Parse logprobs if returned by provider
            token_logprobs: List[TokenLogprob] = []
            if choice.logprobs and choice.logprobs.content:
                for lp_item in choice.logprobs.content:
                    margin: Optional[float] = None
                    top_lps = getattr(lp_item, "top_logprobs", None)
                    if top_lps and len(top_lps) >= 2:
                        top1 = top_lps[0].logprob
                        top2 = top_lps[1].logprob
                        margin = round(top1 - top2, 4)

                    token_str = getattr(lp_item, "token", "")
                    lp_val = getattr(lp_item, "logprob", 0.0)

                    token_logprobs.append(
                        TokenLogprob(
                            token_id=0,
                            token_str=token_str,
                            logprob=round(float(lp_val), 4),
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
                cache_disabled=True,
                thermal_state="nominal",
            )

            return TierResult(
                text=output_text,
                token_ids=[],
                logprobs=token_logprobs,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                timing=timing,
                model_id=self._model_id,
                finish_reason=finish_reason,
                energy_j=0.0,  # Hosted API does not consume local battery energy
                energy_method="none",
                hardware_profile="cloud",
                raw_response_metadata={"hosted": True, "provider_id": getattr(response, "id", "live")},
            )

        except Exception as e:
            # Raise exception immediately: mock mode has been eliminated
            raise RuntimeError(f"Hosted API call to {self._model_id} failed: {e}") from e

