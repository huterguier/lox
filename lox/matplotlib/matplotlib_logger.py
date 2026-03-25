from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.experimental
import jax.numpy as jnp

from lox import logdict
from lox.loggers.logger import Logger, LoggerState
from lox.utils.typing import Array, Key


@jax.tree_util.register_dataclass
@dataclass
class MatplotlibLoggerState(LoggerState):
    key: Key
    id: Array


class MatplotlibLogger(Logger[MatplotlibLoggerState]):
    states: dict[str, Any]
    create: Callable[[Key], Any]
    plot: Callable[[Any, logdict], Any]

    def __init__(
        self, create: Callable[[Key], Any], plot: Callable[[Any, logdict], Any]
    ):
        super().__init__()
        self.create = create
        self.plot = plot
        self.states = {}

    def init(
        self,
        key: Key,
    ) -> MatplotlibLoggerState:
        def callback(key):
            id = jnp.int32(len(self.states.keys()))
            self.states[str(id)] = self.create(key)
            return id

        id = jax.experimental.io_callback(
            callback,
            jax.ShapeDtypeStruct((), jnp.int32),
            key=key,
        )

        return MatplotlibLoggerState(key=key, id=id)

    def callback(self, logger_state: MatplotlibLoggerState, logs: logdict):
        state = self.states[str(logger_state.id)]
        state = self.plot(state, logs)
        self.states[str(logger_state.id)] = state
