"""Base interface for inference tiers in CAMIR."""

from abc import ABC, abstractmethod
from typing import Optional
from camir.traces.schema import TierResult


class BaseTier(ABC):
    """Abstract interface defining the tier contract.
    
    Both local MLX models and hosted remote APIs implement this interface,
    returning an identical, validated TierResult.
    """

    @property
    @abstractmethod
    def tier_name(self) -> str:
        """Return 'small', 'medium', or 'large'."""
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return model identifier."""
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
        """Generate a completion for the given prompt, returning a complete TierResult."""
        pass
