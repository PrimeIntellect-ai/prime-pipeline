import torch

from model import Transformer, TransformerShard


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(checkpoint_path, device, precision):
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
