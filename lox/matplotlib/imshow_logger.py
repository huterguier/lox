import matplotlib.pyplot as plt

from lox.matplotlib.matplotlib_logger import MatplotlibLogger


class ImshowLogger(MatplotlibLogger):

    def __init__(self, argname: str):
        self.argname = argname

        def create(key):
            del key
            fig, ax = plt.subplots()
            return fig, ax

        def plot(state, logs):
            fig, ax = state
            ax.clear()
            ax.imshow(logs[self.argname][0])
            fig.canvas.draw()
            plt.pause(0.001)
            return fig, ax

        super().__init__(create=create, plot=plot)
