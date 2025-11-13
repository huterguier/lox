import os
import shutil
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jax

from lox.logdict import logdict
from lox.loggers.logger import Logger, LoggerState
from lox.save import save, save_callback
from lox.tap import tap
from lox.typing import Key


@jax.tree_util.register_dataclass
@dataclass
class SaveLoggerState(LoggerState):
    key: jax.Array


class SaveLogger(Logger[SaveLoggerState]):
    """
    Logger for saving data to a specified path using JAX's experimental IO callback.
    """

    def __init__(self, path: str, overwrite: bool = False):
        self.path = path
        self.overwrite = overwrite

    def init(self, key: Key) -> SaveLoggerState:
        def callback(key):
            key_data = jax.random.key_data(key)
            folder_name = str(int(f"{key_data[0]}{key_data[1]}"))
            path = self.path + "/" + folder_name
            if not self.overwrite and os.path.exists(path):
                overwrite = input(
                    f"Path {path} already exists. Overwrite? (y/n): "
                ).lower()
                if overwrite == "y":
                    shutil.rmtree(path)
                else:
                    raise FileExistsError(f"Path {path} already exists.")
            os.makedirs(path)

        jax.debug.callback(callback, ordered=True, key=key)

        return SaveLoggerState(key=key)

    def log(
        self, logger_state: SaveLoggerState, logs: logdict, prefix: str = ""
    ) -> None:
        if prefix:
            logs = logs.prefix(prefix)
        save(logs, self.path, key=logger_state.key)

    def tap(
        self,
        f: Callable,
        logger_state: SaveLoggerState,
        argnames: Optional[Sequence[str]] = None,
        prefix: str = "",
    ) -> Callable:
        def callback(logs: logdict):
            save_callback(logs, self.path, key=logger_state.key)

        return tap(f, callback=callback, argnames=argnames, prefix=prefix)
