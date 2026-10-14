"""Apple Silicon local inference tier using MLX-LM.

Supports 4-bit quantized models such as Qwen2.5-1.5B (Small) and Qwen2.5-7B (Medium).
Extracts token-level log-probabilities, top-1/top-2 margins, and high-resolution timing.
"""

import time
from typing import List, Optional
import mlx.core as mx
from camir.accounting.cost import calculate_token_cost
from camir.accounting.energy import EnergyTracker
from camir.accounting.timing import compute_timing_breakdown
from camir.tiers.base import BaseTier
from camir.traces.schema import TierResult, TimingBreakdown, TokenLogprob


class LocalMLXTier(BaseTier):
    """Local tier running 4-bit quantized LLMs on Apple Silicon via MLX."""

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        tier_name: str = "small",
    ):
        self._tier_name = tier_name
        self._model_id = model_id
        self._model = None
        self._tokenizer = None
        self._is_warmed_up = False
        self._load_time_s = 0.0

    @property
    def tier_name(self) -> str:
        return self._tier_name

    @property
    def model_id(self) -> str:
        return self._model_id

    def load_model(self):
        """Lazy load model and tokenizer, isolating load latency."""
        if self._model is not None and self._tokenizer is not None:
            return

        t0 = time.perf_counter()
        import mlx_lm
        self._model, self._tokenizer = mlx_lm.load(self._model_id)
        self._load_time_s = time.perf_counter() - t0

    def warmup(self):
        """Perform a single throwaway prompt to compile Metal shaders and allocate buffers."""
        if self._is_warmed_up:
            return
        self.load_model()
        import mlx_lm
        # Discard the output and timings of this first run
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
        """Generate tokens step-by-step, recording logprobs, margins, and timings."""
        self.load_model()
        if not self._is_warmed_up:
            self.warmup()

        mx.random.seed(seed)

        # Format prompt with system message if tokenizer supports chat templates
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

        # Collect EOS token IDs
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

        with EnergyTracker() as energy_ctx:
            t_start = time.perf_counter()

            # Step-by-step generator yielding (token, logprobs)
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

                # Record Time To First Token (TTFT)
                if first_token_time is None:
                    first_token_time = step_now
                    t_ttft_s = first_token_time - t_start
                    t_prefill_s = t_ttft_s

                tok_id = int(token.item() if hasattr(token, "item") else token)

                # Check EOS
                if tok_id in eos_ids:
                    break

                generated_tokens.append(tok_id)

                # Extract token logprob and top-2 margin
                logprob_val = 0.0
                margin_val: Optional[float] = None

                if step_logprobs is not None:
                    try:
                        mx.eval(step_logprobs)
                        lp_scalar = step_logprobs[tok_id]
                        logprob_val = float(lp_scalar.item() if hasattr(lp_scalar, "item") else lp_scalar)

                        # Top-2 logits/logprobs for margin
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
        )

        return TierResult(
            text=full_output_text.strip(),
            token_ids=generated_tokens,
            logprobs=token_logprobs,
            prompt_tokens=prompt_len,
            output_tokens=output_token_count,
            timing=timing,
            model_id=self._model_id,
            finish_reason="stop" if len(generated_tokens) < max_tokens else "length",
            energy_j=energy_ctx.energy_j,
        )
