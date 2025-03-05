# From: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.distributed as dist
import torch._dynamo.config
import torch._inductor.config
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True 
torch._functorch.config.enable_autograd_cache = True

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import Transformer
from tokenizer import get_tokenizer

def multinomial_sample_one_no_sync(probs_sort: Tensor) -> Tensor:
    """Sample from a multinomial distribution without a cuda synchronization from (B, S) to (B, 1)"""
    # Draw exponential random variables
    q = torch.empty_like(probs_sort).exponential_(1)
    # Select the argmax
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits: Tensor, temperature: float = 1.0, top_k: Optional[int] = None) -> Tensor:
    """Convert logits to probabilities from (B, S, H) to (B, S)"""
    # Temperature scaling (capped at minimum temperature of 1e-5)
    logits = logits / max(temperature, 1e-5)

    # Top-k sampling
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    # Softmax
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits: Tensor, temperature: float = 1.0, top_k: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """Sample from a multinomial distribution without a cuda synchronization"""
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def roundup(val: float, multiplier: float) -> int:
    """Round up to the nearest multiple of multiplier"""
    return ((val - 1) // multiplier + 1) * multiplier

def causal_mask(b, h, q, kv):
    return q >= kv

def prefill(model: Transformer, x: Optional[Tensor], input_pos: Tensor, **sampling_kwargs) -> Optional[Tensor]:
    """Prefill the model with the given input """
    # input_pos: [B, S]
    mask = create_block_mask(causal_mask, 1, 1, input_pos.shape[0], model.max_seq_length, device=x.device)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank != 0: # all but first rank
        # print(f"[{rank}] waiting for input")
        # shape (batch_size, num_prompt_tokens, hidden_size)
        hidden_states = torch.empty((x.size(0), x.size(1), 4096), device=x.device, dtype=torch.bfloat16)
        dist.recv(hidden_states, src=(rank-1) % world_size)
        # print(f"[{rank}] received hidden_states")
        # print(f"{hidden_states=}")
        x = hidden_states

    # print(f"[{rank}] forward")
    output = model(mask, x, input_pos)
    # print(f"[{rank}] done forward {output.shape}")
    # print(f"{output=}")

    if rank != world_size - 1: # all but last rank
        # print(f"[{rank}] sending output")
        dist.send(output, dst=(rank+1) % world_size)
        next_token = torch.empty(size=(x.size(0), 1), dtype=x.dtype, device=x.device)
        dist.recv(next_token, src=1)
    else:
        next_token = sample(output, **sampling_kwargs)[0]
        if dist.get_world_size() > 1:
            dist.send(next_token, dst=0)
    return next_token


def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, block_mask: BlockMask, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode one token"""
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, model.max_seq_length)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank != 0: # all but first rank
        print(f"[{rank}] waiting for input")
        hidden_states = torch.empty((x.size(0), 1, 4096), device=x.device, dtype=torch.bfloat16)
        dist.recv(hidden_states, src=0)
        print(f"[{rank}] received hidden_states {hidden_states.shape}")
        print(f"{hidden_states=}")
        x = hidden_states

    print(f"[{rank}] forward")
    output = model(mask, x, input_pos)
    print(f"[{rank}] done forward {output.shape}")
    print(f"{output=}")

    if rank != world_size - 1: # all but last rank
        print(f"[{rank}] sending output")
        dist.send(output, dst=1)
        next_token = torch.empty(size=(x.size(0), 1), dtype=x.dtype, device=x.device)
        print(f"[{rank}] receiving next_token {next_token.shape}")
        dist.recv(next_token, src=1)
    else:
        next_token = sample(output, **sampling_kwargs)[0]
        print(f"[{rank}] sending next_token {next_token.shape} with {next_token.dtype} and is {next_token}")
        if dist.get_world_size() > 1:
            dist.send(next_token, dst=0)
    return next_token

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, **sampling_kwargs):
    """Decode n tokens"""
    block_mask = create_block_mask(causal_mask, 1, 1, model.max_seq_length, model.max_seq_length, device=cur_token.device)
    new_tokens = []
    for i in range(num_new_tokens):
        next_token = decode_one_token(
            model, cur_token, input_pos, block_mask, **sampling_kwargs
        )
        input_pos += 1
        new_tokens.append(next_token.clone())
        cur_token = next_token.clone()

    return new_tokens

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: Tensor,
    max_new_tokens: int,
    batch_size: int,
    **sampling_kwargs
) -> torch.Tensor:
    # Get number of tokens in prompt and to generate
    num_prompt_tokens = prompt.size(-1)
    num_new_tokens = max_new_tokens
    seq_length = min(num_prompt_tokens + num_new_tokens, model.config.block_size) # Cap at model context

    # Setup cache
    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=seq_length)

    # Create empty tensor of prompt tokens
    prompt = prompt.view(1, -1).repeat(batch_size, 1) # Repeat prompt for each batch
    empty = torch.empty(batch_size, seq_length, dtype=dtype, device=device) # Create empty tensor
    empty[:, :num_prompt_tokens] = prompt # Fill in prompt tokens
    input_pos = torch.arange(num_prompt_tokens, device=device)

    # Prefill prompt tokens
    seq = empty
    print(f"[{dist.get_rank()}] running prefill")
    next_token = prefill(model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs)
    print(f"[{dist.get_rank()}] done prefilling")
    print(f"[{dist.get_rank()}] {next_token=}")

    seq[:, num_prompt_tokens] = next_token.squeeze()

    # Decode remaining tokens
    input_pos = torch.tensor([num_prompt_tokens], device=device, dtype=torch.int)
    generated_tokens = decode_n_tokens(model, next_token.view(batch_size, -1), input_pos, max_new_tokens - 1, **sampling_kwargs)
    seq[:, num_prompt_tokens+1:] = torch.cat(generated_tokens, dim=-1)

    return seq

def encode_tokens(tokenizer, string, bos, device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def load_model(checkpoint_path, device, precision):
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    model = model.to(device=device, dtype=precision)
    return model.eval()

def get_model_size(model):
    model_size = 0
    params = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
            params += sum(
                [
                    p.numel()
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
    return model_size, params

B_INST, E_INST = "[INST]", "[/INST]"

def main(
    prompt: str = "Hello, my name is",
    num_samples: int = 5,
    max_new_tokens: int = 100,
    batch_size: int = 1,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
    compile: bool = True,
    compile_prefill: bool = False,
    precision: torch.dtype = torch.bfloat16,
) -> None:
    # Initialize distributed process group first
    dist.init_process_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"{rank=}", f"{world_size=}")

    # Set seeds for reproducibility across all processes
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    # Optional: Make cuda operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Check if model exists
    assert checkpoint_path.is_file(), checkpoint_path

    # Check if tokenizer exists
    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    assert torch.cuda.is_available(), "CUDA is not available"
    assert torch.cuda.device_count() >= world_size, f"Only {torch.cuda.device_count()} CUDA devices found, but {world_size} are required"
    device = f"cuda:{rank}"

    # Load model
    print(f"Using device={device}")
    print(f"Using precision={precision}")

    # Load model
    print("Loading model ...", end="\r")
    t0 = time.time()
    model = load_model(checkpoint_path, device, precision)
    device_sync(device=device) 
    print(f"Loaded model in {time.time() - t0:.02f} seconds")

    # Shard model
    if world_size > 1:
        from model import shard_model
        model = shard_model(model, rank, world_size)

    # Load tokenizer
    print("Loading tokenizer ...", end="\r")
    t0 = time.time()
    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)
    print(f"Loaded tokenizer in {time.time() - t0:.02f} seconds")

    # Encode prompt
    encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

    torch.manual_seed(1234)
    model_size, params = get_model_size(model)
    if compile:
        global create_block_mask
        create_block_mask = torch.compile(create_block_mask)

        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead") # fullgraph=True

        # Uncomment to squeeze more perf out of prefill
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)


    aggregate_metrics = {'tokens_per_sec': []}
    start = -1 if compile else 0
    for i in range(start, num_samples):
        device_sync(device=device)
        t0 = time.perf_counter()
        decoded = generate(
            model,
            encoded,
            max_new_tokens,
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k,
        )
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        device_sync(device=device)
        t = time.perf_counter() - t0

        # Just displaying the first generation
        print("\n" + "="*10)
        for i in range(batch_size):
            print(tokenizer.decode(decoded[i].tolist()))
            print("="*10 + "\n")
        tokens_generated = decoded.size(-1) - encoded.size(-1)
        generated_tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(generated_tokens_sec)
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {generated_tokens_sec:.02f} tokens/sec")
        print(f"Bandwidth achieved: {model_size * generated_tokens_sec / 1e9:.02f} GB/s")
        total_tokens_sec = decoded.numel() / t
        print(f"FLOPS achieved: {params * total_tokens_sec * 2 / 1e12:.02f} TF/s")
        print()

    print(f"Batch Size: {batch_size}")
    print(f"Prompt Length: {encoded.size(-1)}")
    print(f"Generated tokens: {max_new_tokens}")
    print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    dist.destroy_process_group()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', type=Path, required=True, help='Model checkpoint path.')
    parser.add_argument('--prompt', type=str, default="Hello, my name is", help="Input prompt.")
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')

    args = parser.parse_args()
    main(
        args.prompt, args.num_samples, args.max_new_tokens, args.batch_size, args.top_k,
        args.temperature, args.checkpoint_path, args.compile, args.compile_prefill
    )
