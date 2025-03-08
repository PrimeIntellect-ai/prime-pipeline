from pathlib import Path

import torch
import torch.distributed as dist

from model import Transformer, TransformerShard


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str) -> torch.device:
    if device == "cuda":
        return torch.device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}")
    elif device == "cpu":
        return torch.device("cpu")
    else:
        raise NotImplementedError(f"Device {device} not implemented")


def get_precision(precision: str) -> torch.dtype:
    if precision == "float16":
        return torch.float16
    elif precision == "bfloat16":
        return torch.bfloat16
    else:
        raise NotImplementedError(f"Precision {precision} not implemented")


def load_model(
    checkpoint_path: Path, device: torch.device, precision: torch.dtype
) -> Transformer:
    with torch.device("meta"):
        model = Transformer.from_name(checkpoint_path.parent.name)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    model = model.to(device=device, dtype=precision)
    return model.eval()


def shard_model(model: Transformer, rank: int, world_size: int) -> TransformerShard:
    return TransformerShard(rank, world_size, model)
