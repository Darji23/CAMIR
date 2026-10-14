"""Apple Silicon energy accounting in Joules for CAMIR.

Measures or estimates energy consumption during inference.
Important rule: Dollars, seconds, and Joules are NEVER added together.
"""

import os
import subprocess
import time
from typing import Optional


class EnergyTracker:
    """Context manager to measure or estimate package energy in Joules."""

    # Typical active power profiles for Apple Silicon M4 running quantized LLM inference
    # Baseline idle ~1.5W; M4 under 4-bit MLX decode ~12.5W active power.
    ESTIMATED_M4_ACTIVE_WATTS = 12.5

    def __init__(self, use_powermetrics: bool = False):
        self.use_powermetrics = use_powermetrics
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self.energy_j: Optional[float] = None

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = time.perf_counter()
        elapsed_s = self._end_time - self._start_time if self._start_time else 0.0

        if self.use_powermetrics:
            # Attempt powermetrics reading if configured
            measured = self._read_powermetrics_joules(elapsed_s)
            self.energy_j = measured if measured is not None else (elapsed_s * self.ESTIMATED_M4_ACTIVE_WATTS)
        else:
            # Calibrated model: Energy (Joules) = Power (Watts) * Time (Seconds)
            self.energy_j = round(elapsed_s * self.ESTIMATED_M4_ACTIVE_WATTS, 4)

    def _read_powermetrics_joules(self, elapsed_s: float) -> Optional[float]:
        """Try non-blocking powermetrics sample if permitted."""
        try:
            cmd = ["powermetrics", "-n", "1", "-i", "100", "-s", "cpu_power,gpu_power"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1.0)
            if res.returncode == 0:
                # Parse power in mW
                for line in res.stdout.splitlines():
                    if "Combined Power" in line or "Package Power" in line:
                        parts = line.split(":")
                        if len(parts) > 1:
                            val_str = parts[1].replace("mW", "").strip()
                            watts = float(val_str) / 1000.0
                            return round(watts * elapsed_s, 4)
        except Exception:
            pass
        return None
