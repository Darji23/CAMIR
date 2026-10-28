"""High-resolution timing utilities for CAMIR inference tiers.

Enforces timing rules from proposal:
- Always use time.perf_counter()
- Keep model load time separate from inference latency
- Track TTFT (time to first token) and TPOT (time per output token)
"""

import time
from typing import Optional
from camir.traces.schema import TimingBreakdown


class HighResTimer:
    """Timer using perf_counter for microsecond precision."""

    def __init__(self):
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def start(self):
        self._start = time.perf_counter()
        self._end = None

    def stop(self) -> float:
        self._end = time.perf_counter()
        if self._start is None:
            return 0.0
        return self._end - self._start

    @property
    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        current = self._end if self._end is not None else time.perf_counter()
        return current - self._start


def compute_timing_breakdown(
    t_load_s: float,
    t_prefill_s: float,
    t_ttft_s: float,
    t_decode_s: float,
    output_tokens: int,
    cache_disabled: bool = True,
    thermal_state: str = "nominal",
) -> TimingBreakdown:
    """Compute normalized timing metrics."""
    t_total_s = t_prefill_s + t_decode_s
    tpot_s = t_decode_s / max(1, output_tokens - 1) if output_tokens > 1 else t_decode_s

    return TimingBreakdown(
        t_load_s=max(0.0, t_load_s),
        t_prefill_s=max(0.0, t_prefill_s),
        t_ttft_s=max(0.0, t_ttft_s),
        t_decode_s=max(0.0, t_decode_s),
        t_total_s=max(0.0, t_total_s),
        tpot_s=max(0.0, tpot_s),
        cache_disabled=cache_disabled,
        thermal_state=thermal_state,
    )
