import pickle
from abc import ABC, abstractmethod

import torch


class Serializer(ABC):
    @abstractmethod
    def serialize(self, tensor: torch.Tensor) -> bytes:
        """Serializes a tensor to bytes"""
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> torch.Tensor:
        """Loads serialized bytes to a tensor"""
        pass


class PickleSerializer(Serializer):
    def serialize(self, tensor: torch.Tensor) -> bytes:
        return pickle.dumps(tensor)

    def deserialize(self, data: bytes) -> torch.Tensor:
        return pickle.loads(data)


def get_serializer() -> Serializer:
    return PickleSerializer()
