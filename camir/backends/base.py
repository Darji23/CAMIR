"""Base abstract backend interface for CAMIR.

Decouples inference execution engine (MLX-LM, llama.cpp, vLLM) from
the higher-level Tier abstractions.
"""

from abc import ABC, abstractmethod
from typing import Optional
from camir.traces.schema import TierResult


class BaseBackend(ABC):
    """Abstract interface defining execution engine contracts."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the inference backend (e.g. 'mlx', 'llama.cpp', 'vllm')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if dependencies and runtime requirements for this backend are met."""
        pass

    @abstractmethod
    def load_model(self, model_id: str) -> None:
        """Load and initialize model weights."""
        pass

    @abstractmethod
    def warmup(self) -> None:
        """Perform throwaway pre-run to warm up memory/buffers/shaders."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 42,
        system_prompt: Optional[str] = None,
    ) -> TierResult:
        """Generate response and return structured TierResult."""
        pass
