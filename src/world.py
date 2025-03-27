from typing import Optional


class World:
    def __init__(self, rank: Optional[int] = None, local_rank: Optional[int] = None, size: Optional[int] = None):
        self.rank = rank if rank is not None else 0
        self.local_rank = local_rank if local_rank is not None else self.rank
        self.size = size if size is not None else 1

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


def get_world() -> World:
    global _WORLD
    if _WORLD is None:
        _WORLD = World()
    return _WORLD


def setup_world(rank: int, local_rank: int, size: int) -> None:
    global _WORLD
    _WORLD = World(rank, local_rank, size)
    return _WORLD
