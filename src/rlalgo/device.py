from abc import abstractmethod
from typing import Protocol, Self

from torch import device


class Dev(Protocol):
    '''Provides `to_dev` method.'''
    @abstractmethod
    def to_dev(self, device: device) -> Self:
        raise NotImplementedError

