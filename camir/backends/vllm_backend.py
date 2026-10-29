"""Optional vLLM execution backend stub for CAMIR.

Provides portability for high-throughput GPU serving environments.
"""

from typing import Optional
from camir.backends.base import BaseBackend
from camir.traces.schema import TierResult
from camir.utils.hardware import get_hardware_profile


class VLLMBackend(BaseBackend):
    """Optional backend stub for vLLM engine."""

    def __init__(self, model_id: Optional[str] = None):
        self._model_id = model_id
        self._llm = None

    @property
    def name(self) -> str:
        return "vllm"

    def is_available(self) -> bool:
        try:
            import vllm
            return True
        except ImportError:
            return False

    def load_model(self, model_id: str) -> None:
        self._model_id = model_id
        if not self.is_available():
            return
        import vllm
        self._llm = vllm.LLM(model=model_id)

    def warmup(self) -> None:
        pass

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 42,
        system_prompt: Optional[str] = None,
    ) -> TierResult:
        if not self.is_available():
            raise NotImplementedError(
                "VLLMBackend is an optional portability stub. "
                "Install 'vllm' on Linux/CUDA environments to enable vLLM batch execution."
            )
        raise NotImplementedError("vLLM execution is not yet initialized.")
