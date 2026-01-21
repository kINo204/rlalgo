import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Any
from contextlib import contextmanager

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

logger: Logger

@contextmanager
def logging(mode: str | None = None):
    global logger
    logger = Logger()
    yield
    if not mode:
        print(logger.data)
    elif mode == 'plot':
        _, ax = plt.subplots(1,1)
        logger.plot(ax)
        plt.show(block=True)

def log(*args):
    global logger
    logger.log(*args)
