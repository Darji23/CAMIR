"""Apple Silicon inference backend using MLX-LM.

Provides step-by-step token generation, exact logprobs, top-2 margins,
high-resolution timings (TTFT, prefill, decode), and energy accounting.
"""

import time
from typing import List, Optional
from camir.accounting.energy import EnergyTracker
from camir.accounting.timing import compute_timing_breakdown
from camir.backends.base import BaseBackend
from camir.traces.schema import TierResult, TokenLogprob
from camir.utils.hardware import get_hardware_profile, get_thermal_state


class MLXBackend(BaseBackend):
    """Inference engine powered by MLX and mlx-lm on Apple Silicon."""

    def __init__(self, model_id: Optional[str] = None):
        self._model_id = model_id
        self._model = None
        self._tokenizer = None
        self._is_warmed_up = False
        self._load_time_s = 0.0

    @property
    def name(self) -> str:
        return "mlx"

    def is_available(self) -> bool:
        try:
            import mlx.core as mx
            import mlx_lm
            return True
        except ImportError:
            return False

    def load_model(self, model_id: str) -> None:
        """Lazy load model and tokenizer on Apple Silicon unified memory."""
        if self._model is not None and self._tokenizer is not None and self._model_id == model_id:
            return

        self._model_id = model_id
        t0 = time.perf_counter()
        import mlx_lm
        self._model, self._tokenizer = mlx_lm.load(model_id)
        self._load_time_s = time.perf_counter() - t0

    def warmup(self) -> None:
        """Perform a single throwaway prompt to compile Metal shaders."""
        if self._is_warmed_up or self._model is None or self._tokenizer is None:
            return
        import mlx_lm
        _ = mlx_lm.generate(
            self._model,
            self._tokenizer,
            prompt="Hello",
            max_tokens=2,
            verbose=False,
        )
        self._is_warmed_up = True

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 42,
        system_prompt: Optional[str] = None,
    ) -> TierResult:
        if not self.is_available():
            raise RuntimeError("MLXBackend is not available on this platform or mlx is not installed.")

        if self._model is None:
            if not self._model_id:
                raise ValueError("No model_id loaded in MLXBackend. Call load_model() first.")
            self.load_model(self._model_id)

        if not self._is_warmed_up:
            self.warmup()

        import mlx.core as mx

        mx.random.seed(seed)

        if system_prompt and hasattr(self._tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            try:
                formatted_prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            formatted_prompt = prompt

        try:
            from mlx_lm.generate import generate_step
        except ImportError:
            from mlx_lm.utils import generate_step

        try:
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=temperature)
        except Exception:
            sampler = None

        prompt_tokens = self._tokenizer.encode(formatted_prompt)
        prompt_len = len(prompt_tokens)
        prompt_tokens_mx = mx.array(prompt_tokens)

        eos_ids = set()
        if hasattr(self._tokenizer, "eos_token_ids") and self._tokenizer.eos_token_ids:
            eos_ids.update(self._tokenizer.eos_token_ids)
        if hasattr(self._tokenizer, "eos_token_id") and self._tokenizer.eos_token_id is not None:
            eos_ids.add(self._tokenizer.eos_token_id)

        generated_tokens: List[int] = []
        token_logprobs: List[TokenLogprob] = []

        t_prefill_s = 0.0
        t_ttft_s = 0.0
        t_decode_s = 0.0
        first_token_time: Optional[float] = None

        with EnergyTracker(profile="apple_silicon") as energy_ctx:
            t_start = time.perf_counter()

            kwargs = {"max_tokens": max_tokens}
            if sampler is not None:
                kwargs["sampler"] = sampler

            generator = generate_step(
                prompt=prompt_tokens_mx,
                model=self._model,
                **kwargs,
            )

            for token, step_logprobs in generator:
                step_now = time.perf_counter()

                if first_token_time is None:
                    first_token_time = step_now
                    t_ttft_s = first_token_time - t_start
                    t_prefill_s = t_ttft_s

                tok_id = int(token.item() if hasattr(token, "item") else token)

                if tok_id in eos_ids:
                    break

                generated_tokens.append(tok_id)

                logprob_val = 0.0
                margin_val: Optional[float] = None

                if step_logprobs is not None:
                    try:
                        mx.eval(step_logprobs)
                        lp_scalar = step_logprobs[tok_id]
                        logprob_val = float(lp_scalar.item() if hasattr(lp_scalar, "item") else lp_scalar)

                        top_indices = mx.argpartition(-step_logprobs, kth=2)[:2]
                        mx.eval(top_indices)
                        top1_idx = int(top_indices[0].item() if hasattr(top_indices[0], "item") else top_indices[0])
                        top2_idx = int(top_indices[1].item() if hasattr(top_indices[1], "item") else top_indices[1])

                        l1_scalar = step_logprobs[top1_idx]
                        l2_scalar = step_logprobs[top2_idx]
                        l1 = float(l1_scalar.item() if hasattr(l1_scalar, "item") else l1_scalar)
                        l2 = float(l2_scalar.item() if hasattr(l2_scalar, "item") else l2_scalar)
                        if l2 > l1:
                            l1, l2 = l2, l1
                        margin_val = round(l1 - l2, 4)
                    except Exception:
                        pass

                token_str = self._tokenizer.decode([tok_id])
                token_logprobs.append(
                    TokenLogprob(
                        token_id=tok_id,
                        token_str=token_str,
                        logprob=round(logprob_val, 4),
                        top2_margin=margin_val,
                    )
                )

                if len(generated_tokens) >= max_tokens:
                    break

            t_end = time.perf_counter()

        if first_token_time is not None:
            t_decode_s = t_end - first_token_time
        else:
            t_decode_s = 0.0

        full_output_text = self._tokenizer.decode(generated_tokens)
        output_token_count = len(generated_tokens)

        timing = compute_timing_breakdown(
            t_load_s=self._load_time_s,
            t_prefill_s=t_prefill_s,
            t_ttft_s=t_ttft_s,
            t_decode_s=t_decode_s,
            output_tokens=output_token_count,
            cache_disabled=True,
            thermal_state=get_thermal_state(),
        )

        return TierResult(
            text=full_output_text.strip(),
            token_ids=generated_tokens,
            logprobs=token_logprobs,
            prompt_tokens=prompt_len,
            output_tokens=output_token_count,
            timing=timing,
            model_id=self._model_id or "mlx-local",
            finish_reason="stop" if len(generated_tokens) < max_tokens else "length",
            energy_j=energy_ctx.energy_j,
            energy_method=energy_ctx.energy_method,
            hardware_profile="apple_silicon",
        )
