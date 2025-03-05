import sys
import torch
from world import World
from model import Transformer

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_world():
    return World()

def setup_logger(rank: int, console_level: str = "INFO", file_level: str = "DEBUG"):
    from loguru import logger
    console_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> [<level>{level}</level>] [<level>Rank {extra[rank]}</level>] - <level>{message}</level>"
    file_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> [<level>{level}</level>] [<level>Rank {extra[rank]}</level>] - <level>{message}</level>"
    logger.remove()
    logger.add(f'logs/process_{rank}.log', level=file_level, format=file_format, mode='w')
    logger.add(sys.stdout, level=console_level, format=console_format)
    logger = logger.bind(rank=rank)

    return logger

def load_model(checkpoint_path, device, precision):
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    model = model.to(device=device, dtype=precision)
    return model.eval()
