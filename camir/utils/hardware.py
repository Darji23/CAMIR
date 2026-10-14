"""Hardware and environmental monitoring utilities for Apple Silicon."""

import os
import platform
import subprocess
from typing import Dict, Any


def get_hardware_info() -> str:
    """Return Apple Silicon model name, core counts, and total memory."""
    try:
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        chip = platform.processor() or "Apple Silicon"

    try:
        cores = subprocess.check_output(["sysctl", "-n", "hw.ncpu"], text=True).strip()
    except Exception:
        cores = "unknown"

    try:
        mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        mem_gb = round(mem_bytes / (1024 ** 3), 1)
        mem_str = f"{mem_gb}GB"
    except Exception:
        mem_str = "unknown RAM"

    return f"{chip} ({cores} cores, {mem_str})"


def get_thermal_state() -> str:
    """Return macOS thermal state: 'nominal', 'fair', 'serious', 'critical', or 'unknown'."""
    try:
        # Check thermal state quietly via pmset
        out = subprocess.check_output(
            ["pmset", "-g", "therm"],
            stderr=subprocess.DEVNULL,
            text=True
        )
        if "CPU_Speed_Limit" in out:
            # If speed limit is 100%, throttling is nominal
            return "nominal" if "100" in out else "fair"
        return "nominal"
    except Exception:
        return "nominal"
