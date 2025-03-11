import os


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
