import torch.distributed as dist

class World:
    def __init__(self):
        dist.init_process_group()
        self.rank = dist.get_rank()
        self.size = dist.get_world_size()

    @property
    def is_first_stage(self):
        return self.rank == 0

    @property
    def is_last_stage(self):
        return self.rank == self.size - 1

    def __str__(self):
        return f"World(rank={self.rank}, world_size={self.size})"