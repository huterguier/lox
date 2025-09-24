import jax
from lox.logdict import logdict
from dataclasses import dataclass
from typing import Callable, Sequence, Optional


from .logger import Logger, LoggerState


@jax.tree_util.register_dataclass
@dataclass
class MultiLoggerState(LoggerState):
    logger_states: tuple[LoggerState, ...]


class MultiLogger(Logger[MultiLoggerState]):
    loggers: Sequence[Logger]

    def __init__(self, loggers: Sequence[Logger]):
        self.loggers = loggers

    def init(self, *args, **kwargs) -> MultiLoggerState:
        logger_states = tuple(logger.init(*args, **kwargs) for logger in self.loggers)
        return MultiLoggerState(logger_states=logger_states)

    def log(self, logger_state: MultiLoggerState, logs: logdict):
        for sub_logger, sub_logger_state in zip(
            self.loggers, logger_state.logger_states
        ):
            sub_logger.log(sub_logger_state, logs)

    def tap(
        self,
        f: Callable,
        logger_state: MultiLoggerState,
        argnames: Optional[Sequence[str]] = None,
    ) -> Callable:
        f_tapped = f
        for sub_logger, sub_logger_state in zip(
            self.loggers, logger_state.logger_states
        ):
            f_tapped = sub_logger.tap(f_tapped, sub_logger_state, argnames=argnames)
        return f_tapped
