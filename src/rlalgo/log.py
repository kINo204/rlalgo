from contextlib import contextmanager
from typing import Any

import matplotlib.pyplot as plt


class Logger():
    data: dict[str, list[Any]]

    def __init__(self) -> None:
        self.data = dict()
    
    def log(self, name: str, value):
        if not self.data.get(name):
            self.data[name] = [value]
        else:
            self.data[name].append(value)

    def plot(self) -> None:
        _fig, axes = plt.subplots(len(self.data.keys()),1)
        axes = iter(axes)
        for name, values in self.data.items():
            ax = next(axes)
            ax.plot(values)
            ax.set_ylabel(name)

_logger: Logger

@contextmanager
def logging(tag: str, mode: str | None = None):
    global _logger
    _logger = Logger()
    yield
    if not mode:
        print(_logger.data)
    elif mode == 'plot':
        _logger.plot()
        plt.show(block=True)

def log(*args):
    global _logger
    _logger.log(*args)
