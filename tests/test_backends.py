"""Unit tests for the CAMIR inference backend portability layer."""

import pytest
from camir.backends.base import BaseBackend
from camir.backends.factory import get_backend
from camir.backends.llamacpp_backend import LlamaCppBackend
from camir.backends.mlx_backend import MLXBackend
from camir.backends.vllm_backend import VLLMBackend


def test_backend_instantiation_via_factory():
    mlx = get_backend("mlx")
    assert isinstance(mlx, MLXBackend)
    assert mlx.name == "mlx"

    llama = get_backend("llama.cpp")
    assert isinstance(llama, LlamaCppBackend)
    assert llama.name == "llama.cpp"

    vllm = get_backend("vllm")
    assert isinstance(vllm, VLLMBackend)
    assert vllm.name == "vllm"

    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("nonexistent_backend")


def test_llamacpp_stub_interface():
    stub = LlamaCppBackend(model_path="models/test.gguf")
    assert stub.name == "llama.cpp"
    # When llama_cpp is not installed, generate() should raise informative NotImplementedError
    if not stub.is_available():
        with pytest.raises(NotImplementedError, match="portability stub"):
            stub.generate("Test prompt")


def test_vllm_stub_interface():
    stub = VLLMBackend(model_id="Qwen/Qwen2.5-7B")
    assert stub.name == "vllm"
    if not stub.is_available():
        with pytest.raises(NotImplementedError, match="portability stub"):
            stub.generate("Test prompt")
