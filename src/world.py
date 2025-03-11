import os
from typing import Optional


class World:
    def __init__(self):
        self.rank = int(os.environ.get("RANK", 0))
        self.size = int(os.environ.get("WORLD_SIZE", 1))

    @property
    def is_first_stage(self) -> bool:
        return self.rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self.rank == self.size - 1

    @property
    def first_stage_rank(self) -> int:
        return 0

    @property
    def last_stage_rank(self) -> int:
        return self.size - 1

    @property
    def is_master(self) -> bool:
        return self.rank == 0


_WORLD: Optional[World] = None


def setup_world():
    global _WORLD
    assert _WORLD is None, "World already setup"
    _WORLD = World()


def get_world() -> World:
    global _WORLD
    assert _WORLD is not None, "World not setup"
    return _WORLD
