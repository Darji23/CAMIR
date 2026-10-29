"""Hardware-specific energy accounting plugins for CAMIR.

Measures or estimates energy consumption during inference across different hardware profiles:
- Apple Silicon (via powermetrics when available, otherwise calibrated 12.5 W active model)
- NVIDIA GPUs (via NVML / nvidia-smi if available, otherwise estimated)
- Linux x86 (via Intel/AMD RAPL sysfs counters when available)
- Cloud / Remote (explicitly labeled 'none' or 'estimated')

Important rule: Dollars, seconds, and Joules are NEVER added together.
Energy is always explicitly labeled as 'estimated', 'measured', or 'none'.
"""

from abc import ABC, abstractmethod
import os
import subprocess
import time
from typing import Optional, Tuple
from camir.utils.hardware import get_hardware_profile


class BaseEnergyPlugin(ABC):
    """Abstract interface for hardware energy tracking plugins."""

    @property
    @abstractmethod
    def profile_name(self) -> str:
        """Name of the hardware profile."""
        pass

    @abstractmethod
    def start(self) -> None:
        """Start tracking energy consumption."""
        pass

    @abstractmethod
    def stop(self) -> Tuple[Optional[float], str, str]:
        """Stop tracking and return (energy_joules, energy_method, label)."""
        pass


class AppleSiliconEnergyPlugin(BaseEnergyPlugin):
    """Energy accounting on Apple Silicon using powermetrics or calibrated 12.5 W model."""

    ESTIMATED_ACTIVE_WATTS: float = 12.5

    def __init__(self, use_powermetrics: bool = False):
        self.use_powermetrics = use_powermetrics
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    @property
    def profile_name(self) -> str:
        return "apple_silicon"

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def stop(self) -> Tuple[Optional[float], str, str]:
        self._end_time = time.perf_counter()
        elapsed_s = self._end_time - self._start_time if self._start_time else 0.0

        if self.use_powermetrics:
            measured = self._read_powermetrics(elapsed_s)
            if measured is not None:
                return measured, "measured", f"{measured:.4f} J (measured)"

        estimated_j = round(elapsed_s * self.ESTIMATED_ACTIVE_WATTS, 4)
        label = f"{estimated_j:.4f} J (estimated, {self.ESTIMATED_ACTIVE_WATTS}W)"
        return estimated_j, "estimated", label

    def _read_powermetrics(self, elapsed_s: float) -> Optional[float]:
        try:
            cmd = ["powermetrics", "-n", "1", "-i", "100", "-s", "cpu_power,gpu_power"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1.0)
            if res.returncode == 0:
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


class NvidiaEnergyPlugin(BaseEnergyPlugin):
    """Energy accounting on NVIDIA GPUs via NVML / nvidia-smi."""

    def __init__(self):
        self._start_time: Optional[float] = None
        self._start_energy_mj: Optional[float] = None

    @property
    def profile_name(self) -> str:
        return "nvidia"

    def _get_gpu_energy_mj(self) -> Optional[float]:
        try:
            # Query cumulative energy in mJ via nvidia-smi if supported
            cmd = ["nvidia-smi", "--query-gpu=energy.consumption", "--format=csv,noheader,nounits"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1.0)
            if res.returncode == 0 and res.stdout.strip():
                return float(res.stdout.strip().splitlines()[0])
        except Exception:
            pass
        return None

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._start_energy_mj = self._get_gpu_energy_mj()

    def stop(self) -> Tuple[Optional[float], str, str]:
        end_time = time.perf_counter()
        end_energy_mj = self._get_gpu_energy_mj()

        if self._start_energy_mj is not None and end_energy_mj is not None and end_energy_mj >= self._start_energy_mj:
            joules = round((end_energy_mj - self._start_energy_mj) / 1000.0, 4)
            return joules, "measured", f"{joules:.4f} J (measured, NVML)"

        elapsed_s = end_time - (self._start_time or end_time)
        # Fallback estimation for typical modern accelerator active power (~150W)
        est_joules = round(elapsed_s * 150.0, 4)
        return est_joules, "estimated", f"{est_joules:.4f} J (estimated, 150W GPU)"


class RaplEnergyPlugin(BaseEnergyPlugin):
    """Energy accounting on Linux x86 via Intel/AMD RAPL sysfs powercap."""

    RAPL_PATH = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"

    def __init__(self):
        self._start_energy_uj: Optional[int] = None
        self._start_time: Optional[float] = None

    @property
    def profile_name(self) -> str:
        return "cpu_rapl"

    def _read_rapl(self) -> Optional[int]:
        if os.path.exists(self.RAPL_PATH):
            try:
                with open(self.RAPL_PATH, "r") as f:
                    return int(f.read().strip())
            except Exception:
                pass
        return None

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._start_energy_uj = self._read_rapl()

    def stop(self) -> Tuple[Optional[float], str, str]:
        end_time = time.perf_counter()
        end_energy_uj = self._read_rapl()

        if self._start_energy_uj is not None and end_energy_uj is not None and end_energy_uj >= self._start_energy_uj:
            joules = round((end_energy_uj - self._start_energy_uj) / 1e6, 4)
            return joules, "measured", f"{joules:.4f} J (measured, RAPL)"

        elapsed_s = end_time - (self._start_time or end_time)
        est_joules = round(elapsed_s * 65.0, 4)  # 65W TDP baseline
        return est_joules, "estimated", f"{est_joules:.4f} J (estimated, 65W CPU)"


class CloudEnergyPlugin(BaseEnergyPlugin):
    """Energy accounting for cloud / hosted API tiers where hardware counters are unavailable."""

    @property
    def profile_name(self) -> str:
        return "cloud"

    def start(self) -> None:
        pass

    def stop(self) -> Tuple[Optional[float], str, str]:
        return None, "none", "N/A (cloud API / no direct energy telemetry)"


def get_energy_plugin(
    profile: Optional[str] = None,
    use_powermetrics: bool = False,
) -> BaseEnergyPlugin:
    """Factory function returning the appropriate energy plugin for a hardware profile."""
    prof = (profile or get_hardware_profile()).strip().lower()
    if prof == "apple_silicon":
        return AppleSiliconEnergyPlugin(use_powermetrics=use_powermetrics)
    elif prof == "nvidia":
        return NvidiaEnergyPlugin()
    elif prof in ("cpu_rapl", "generic_cpu"):
        return RaplEnergyPlugin()
    elif prof == "cloud":
        return CloudEnergyPlugin()
    else:
        return AppleSiliconEnergyPlugin(use_powermetrics=use_powermetrics)


class EnergyTracker:
    """Context manager to measure or estimate package energy in Joules.
    
    Delegates to per-profile energy plugins (Apple Silicon, NVIDIA, RAPL, Cloud).
    Every calculated energy is explicitly tagged with its method ('estimated', 'measured', or 'none').
    """

    # Calibrated active power profile for Apple Silicon M4 running 4-bit quantized MLX decode: 12.5 W
    ESTIMATED_M4_ACTIVE_WATTS: float = 12.5

    def __init__(self, profile: Optional[str] = None, use_powermetrics: bool = False):
        self.profile = profile or get_hardware_profile()
        self.use_powermetrics = use_powermetrics
        self.plugin = get_energy_plugin(self.profile, use_powermetrics=use_powermetrics)
        self.energy_j: Optional[float] = None
        self.energy_method: str = "estimated"
        self._label: str = ""

    def __enter__(self):
        self.plugin.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.energy_j, self.energy_method, self._label = self.plugin.stop()

    @property
    def label(self) -> str:
        """Formatted human-readable string with explicit estimated/measured label."""
        if self._label:
            return self._label
        val = f"{self.energy_j:.4f} J" if self.energy_j is not None else "N/A"
        return f"{val} ({self.energy_method})"
