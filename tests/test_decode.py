import autorootcwd  # noqa: F401
import pytest

from src.generate import generate  # noqa: E402
from src.setup import setup  # noqa: E402


@pytest.fixture(params=["torch", "iroh"])
def backend(request):
    return request.param


TOKENS = [1, 15043, 29892, 590, 1024, 338, 2812, 25462]
BATCH_TOKENS = [
    [1, 15043, 29892, 590, 1024, 338, 18007, 29875],
    [1, 15043, 29892, 590, 1024, 338, 5061, 8625],
]


def test_single_node_single_batch(backend: str):
    model, _, prompt_tokens, num_prompt_tokens, _, micro_batch_size, _ = setup(
        rank=0,
        local_rank=0,
        world_size=1,
        log_level="CRITICAL",
        seed=1234,
        device="cuda",
        precision="bfloat16",
        model_name="meta-llama/llama-2-7b-chat-hf",
        prompt="Hello, my name is",
        compile=False,
        backend=backend,
        num_new_tokens=10,
        num_cache_tokens=16,
        num_micro_batches=1,
        batch_size=1,
        micro_batch_size=None,
        dummy=False,
    )

    num_micro_batches = len(prompt_tokens)
    decoded_tokens, _, _ = generate(
        model,
        prompt_tokens,
        num_prompt_tokens=num_prompt_tokens,
        num_new_tokens=2,
        num_micro_batches=num_micro_batches,
        micro_batch_size=micro_batch_size,
        disable_tqdm=True,
    )
    assert decoded_tokens.ndim == 2
    assert decoded_tokens.shape == (1, 8)
    assert decoded_tokens.squeeze().tolist() == TOKENS


def test_single_node_multiple_batches(backend: str):
    model, _, prompt_tokens, num_prompt_tokens, _, micro_batch_size, _ = setup(
        rank=0,
        local_rank=0,
        world_size=1,
        log_level="CRITICAL",
        seed=1234,
        device="cuda",
        precision="bfloat16",
        model_name="meta-llama/llama-2-7b-chat-hf",
        prompt="Hello, my name is",
        compile=False,
        backend=backend,
        num_new_tokens=10,
        num_cache_tokens=16,
        num_micro_batches=2,
        batch_size=2,
        micro_batch_size=None,
        dummy=False,
    )

    decoded_tokens, _, _ = generate(
        model,
        prompt_tokens,
        num_prompt_tokens=num_prompt_tokens,
        num_new_tokens=2,
        num_micro_batches=2,
        micro_batch_size=micro_batch_size,
        disable_tqdm=True,
    )
    assert decoded_tokens.ndim == 2
    assert decoded_tokens.shape == (2, 8)
    assert decoded_tokens[0].tolist() == BATCH_TOKENS[0]
    assert decoded_tokens[1].tolist() == BATCH_TOKENS[1]
