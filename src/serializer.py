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
    def __init__(self, device: torch.device):
        self.device = device

    def serialize(self, tensor: torch.Tensor) -> bytes:
        return pickle.dumps(tensor.cpu())

    def deserialize(self, data: bytes) -> torch.Tensor:
        return pickle.loads(data).to(self.device)


def get_serializer(device: torch.device) -> Serializer:
    return PickleSerializer(device)
