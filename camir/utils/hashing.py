"""Deterministic hashing utilities for CAMIR trace indexing.

Guarantees reproducible cache keys by hashing:
- model
- quantization
- max tokens
- temperature
- prompt template (or system prompt format)
- library versions (mlx, python, etc.)
- seed
"""

import hashlib
import json
import platform
from typing import Any, Dict, Optional
from camir.utils.hardware import get_hardware_profile


def get_library_versions() -> Dict[str, str]:
    """Capture runtime package versions for reproducibility."""
    versions = {
        "python": platform.python_version(),
    }
    for pkg in ["mlx", "mlx_lm", "openai", "duckdb", "pydantic"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = str(getattr(mod, "__version__", "unknown"))
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute a canonical SHA256 hash of generation parameters.
    
    Covers model, quantization, max tokens, temperature, prompt template,
    seed, library versions, and hardware_profile.
    """
    raw_versions = config.get("library_versions")
    if raw_versions is None:
        lib_versions = get_library_versions()
    elif isinstance(raw_versions, dict):
        lib_versions = {str(k): str(v) for k, v in raw_versions.items()}
    else:
        lib_versions = {"version_str": str(raw_versions)}

    # Canonicalize prompt template
    prompt_template = str(
        config.get("prompt_template")
        or config.get("system_prompt")
        or "standard_instruct"
    )

    hardware_profile = str(
        config.get("hardware_profile") or get_hardware_profile()
    ).strip().lower()

    normalized = {
        "model_id": str(config.get("model_id", "")).strip().lower(),
        "quantization": str(config.get("quantization", "none")).strip().lower(),
        "max_tokens": int(config.get("max_tokens", 512)),
        "temperature": float(config.get("temperature", 0.0)),
        "seed": int(config.get("seed", 42)),
        "prompt_template": prompt_template,
        "hardware_profile": hardware_profile,
        "library_versions": lib_versions,
    }
    encoded = json.dumps(normalized, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]

