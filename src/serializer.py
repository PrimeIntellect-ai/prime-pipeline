import pickle
from abc import ABC, abstractmethod

import torch


class Serializer(ABC):
    @abstractmethod
    def serialize(self, tensor: torch.Tensor) -> bytes:
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> torch.Tensor:
        pass


class PickleSerializer(Serializer):
    def serialize(self, tensor: torch.Tensor) -> bytes:
        return pickle.dumps(tensor.cpu())

    def deserialize(self, data: bytes) -> torch.Tensor:
        return pickle.loads(data)


def get_serializer() -> Serializer:
    return PickleSerializer()
