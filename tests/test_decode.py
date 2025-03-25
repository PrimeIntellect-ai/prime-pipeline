import autorootcwd  # noqa: F401
import pytest

from src.generate import generate  # noqa: E402
from src.setup import setup  # noqa: E402


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    return request.param


@pytest.fixture(params=["torch", "iroh"])
def backend(request):
    return request.param


CPU_TOKENS = [1, 15043, 29892, 590, 1024, 338, 2259, 322, 306, 626, 263, 29871, 29941, 29900, 29899, 6360]
CUDA_TOKENS = [1, 15043, 29892, 590, 1024, 338, 518, 1170, 1402, 322, 306, 626, 263, 518, 29928, 11124]


def test_single_node_single_batch(backend: str, device: str):
    model, _, prompt_tokens, micro_batch_size = setup(
        rank=0,
        world_size=1,
        log_level="CRITICAL",
        seed=1234,
        device=device,
        precision="bfloat16",
        model_name="meta-llama/llama-2-7b-chat-hf",
        prompt="Hello, my name is",
        compile=False,
        backend=backend,
        num_micro_batches=1,
        batch_size=1,
        dummy=False,
    )

    decoded_tokens, _, _ = generate(model, prompt_tokens, num_new_tokens=3, micro_batch_size=micro_batch_size)
    assert decoded_tokens.ndim == 2
    assert decoded_tokens.shape == (1, 9)
    assert decoded_tokens.squeeze().tolist() == CPU_TOKENS[:9] if device == "cpu" else CUDA_TOKENS[:9]


BATCH_CPU_TOKENS = [[1, 15043, 29892, 590, 1024, 338, 2259, 29892, 322], [1, 15043, 29892, 590, 1024, 338, 9937, 322, 306]]
BATCH_CUDA_TOKENS = [[1, 15043, 29892, 590, 1024, 338, 2259, 29892, 322], [1, 15043, 29892, 590, 1024, 338, 9937, 322, 306]]


def test_single_node_multiple_batches(backend: str, device: str):
    model, _, prompt_tokens, micro_batch_size = setup(
        rank=0,
        world_size=1,
        log_level="CRITICAL",
        seed=1234,
        device=device,
        precision="bfloat16",
        model_name="meta-llama/llama-2-7b-chat-hf",
        prompt="Hello, my name is",
        compile=False,
        backend=backend,
        num_micro_batches=1,
        batch_size=2,
        dummy=False,
    )

    decoded_tokens, _, _ = generate(model, prompt_tokens, num_new_tokens=3, micro_batch_size=micro_batch_size)
    assert decoded_tokens.ndim == 2
    assert decoded_tokens.shape == (2, 9)
    assert decoded_tokens[0].tolist() == BATCH_CPU_TOKENS[0] if device == "cpu" else BATCH_CUDA_TOKENS[0]
    assert decoded_tokens[1].tolist() == BATCH_CPU_TOKENS[1] if device == "cpu" else BATCH_CUDA_TOKENS[1]
