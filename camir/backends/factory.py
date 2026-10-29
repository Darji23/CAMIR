"""Factory for inference execution backends."""

from typing import Optional
from camir.backends.base import BaseBackend
from camir.backends.mlx_backend import MLXBackend
from camir.backends.llamacpp_backend import LlamaCppBackend
from camir.backends.vllm_backend import VLLMBackend


def get_backend(name: str = "mlx", model_id: Optional[str] = None) -> BaseBackend:
    """Instantiate and return the requested inference backend.
    
    Supported: 'mlx', 'llama.cpp', 'vllm'.
    """
    clean_name = name.lower().replace("-", "").replace("_", "").replace(".", "")
    if clean_name in ("mlx", "mlxlocal"):
        return MLXBackend(model_id=model_id)
    elif clean_name in ("llamacpp", "llama", "gguf"):
        return LlamaCppBackend(model_path=model_id)
    elif clean_name == "vllm":
        return VLLMBackend(model_id=model_id)

    else:
        raise ValueError(f"Unknown backend '{name}'. Available: 'mlx', 'llama.cpp', 'vllm'")
