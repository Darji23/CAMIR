"""Deterministic hashing utilities for CAMIR trace indexing."""

import hashlib
import json
from typing import Any, Dict


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute a canonical SHA256 hash of generation parameters.
    
    Standardized fields: model_id, temperature, max_tokens, seed, quantization.
    """
    normalized = {
        "model_id": str(config.get("model_id", "")),
        "temperature": float(config.get("temperature", 0.0)),
        "max_tokens": int(config.get("max_tokens", 512)),
        "seed": int(config.get("seed", 42)),
        "quantization": str(config.get("quantization", "none")),
        "system_prompt": str(config.get("system_prompt", "")),
    }
    encoded = json.dumps(normalized, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
