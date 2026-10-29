"""Portability backends for CAMIR.

Provides abstract backend interface and concrete implementations:
- MLX-LM: native Apple Silicon inference
- llama.cpp: cross-platform GGUF stub
- vLLM: high-throughput GPU stub
"""

from camir.backends.base import BaseBackend
from camir.backends.factory import get_backend
from camir.backends.llamacpp_backend import LlamaCppBackend
from camir.backends.mlx_backend import MLXBackend
from camir.backends.vllm_backend import VLLMBackend

__all__ = [
    "BaseBackend",
    "MLXBackend",
    "LlamaCppBackend",
    "VLLMBackend",
    "get_backend",
]
