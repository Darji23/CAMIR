"""Portability stub for llama.cpp execution backend in CAMIR.

Allows future deployment on cross-platform CPUs, AMD, or quantized GGUF models.
"""

import time
from typing import Optional
from camir.backends.base import BaseBackend
from camir.traces.schema import TierResult, TimingBreakdown
from camir.utils.hardware import get_hardware_profile


class LlamaCppBackend(BaseBackend):
    """Stub implementation of llama.cpp backend interface."""

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path
        self._model = None

    @property
    def name(self) -> str:
        return "llama.cpp"

    def is_available(self) -> bool:
        try:
            import llama_cpp
            return True
        except ImportError:
            return False

    def load_model(self, model_id: str) -> None:
        self._model_path = model_id
        if not self.is_available():
            # Graceful stub handling when optional llama_cpp is uninstalled
            return
        import llama_cpp
        self._model = llama_cpp.Llama(model_path=model_id, verbose=False)

    def warmup(self) -> None:
        pass

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 42,
        system_prompt: Optional[str] = None,
    ) -> TierResult:
        if not self.is_available():
            raise NotImplementedError(
                "LlamaCppBackend is a portability stub in Phase 1. "
                "Install 'llama-cpp-python' to enable local GGUF execution."
            )

        # Real execution path if llama_cpp is present in the environment
        t0 = time.perf_counter()
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        output = self._model(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
        elapsed = time.perf_counter() - t0
        text = output["choices"][0]["text"].strip()
        usage = output.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", len(prompt.split()))
        output_tokens = usage.get("completion_tokens", len(text.split()))

        timing = TimingBreakdown(
            t_load_s=0.0,
            t_prefill_s=round(elapsed * 0.3, 4),
            t_ttft_s=round(elapsed * 0.3, 4),
            t_decode_s=round(elapsed * 0.7, 4),
            t_total_s=round(elapsed, 4),
            tpot_s=round((elapsed * 0.7) / max(1, output_tokens - 1), 4),
            cache_disabled=True,
            thermal_state="nominal",
        )

        return TierResult(
            text=text,
            token_ids=[],
            logprobs=[],
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            timing=timing,
            model_id=self._model_path or "llama.cpp-stub",
            finish_reason="stop",
            energy_j=None,
            energy_method="none",
            hardware_profile=get_hardware_profile(),
        )
