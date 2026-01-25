from contextlib import contextmanager
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class Logger():
    data: dict[str, list[Any]]

    def __init__(self) -> None:
        self.data = dict()
    
    def log(self, name: str, value):
        if not self.data.get(name):
            self.data[name] = [value]
        else:
            self.data[name].append(value)

    def plot(self, ax: Axes) -> None:
        for name, values in self.data.items():
            ax.plot(values, label=name)
        ax.legend()

_logger: Logger

@contextmanager
def logging(tag: str, mode: str | None = None):
    global _logger
    _logger = Logger()
    yield
    if not mode:
        print(_logger.data)
    elif mode == 'plot':
        _, ax = plt.subplots(1,1)
        _logger.plot(ax)
        ax.set_xlabel(tag)
        plt.show(block=True)

def log(*args):
    global _logger
    _logger.log(*args)
