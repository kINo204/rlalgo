from .env import Env
from typing import Protocol
from abc import ABC, abstractmethod

class Algorithm[ModelT](Protocol):
    @abstractmethod
    def train(self, model: ModelT, env: Env) -> None:
        raise NotImplementedError
    