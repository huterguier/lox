from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Generic, Sequence, TypeVar

import jax

from lox.logdict import logdict
from lox.spooling import spool
from lox.tapping import tap
from lox.utils.typing import PyTree


@jax.tree_util.register_dataclass
@dataclass
class LoggerState:
    pass


LoggerStateT = TypeVar("LoggerStateT", bound=LoggerState)


class Logger(Generic[LoggerStateT], ABC):

    @abstractmethod
    def init(self, *args, **kwargs) -> LoggerStateT:
        pass

    def log(self, logger_state: LoggerStateT, logs: logdict) -> LoggerStateT:
        jax.debug.callback(
            self.callback,
            logger_state=logger_state,
            logs=logs,
            ordered=True,
        )
        return logger_state

    @abstractmethod
    def callback(self, logger_state: LoggerStateT, logs: logdict):
        pass

    def spool(
        self,
        f: Callable,
        logger_state: LoggerStateT,
        keep_logs: bool = False,
        interval: int | None = None,
        reduce: str | None = None,
        prefix: str = "",
    ) -> Callable[..., tuple[PyTree, LoggerStateT]]:
        """
        Wraps a function to log its output.

        Args:
            f: The function to be wrapped.
            logger_state: The state of the logger.
            keep_logs: Whether to keep all logs or just the reduced value.
            interval: The interval at which to log.
            reduce: The reduction method to apply to the logs.
            prefix: An optional prefix to add to the log keys.

        Returns:
          A wrapped function that logs its output.
        """

        @wraps(f)
        def wrapped(*args, **kwargs):
            y, logs = spool(
                f,
                keep_logs=keep_logs,
                interval=interval,
                reduce=reduce,
                prefix=prefix,
            )(*args, **kwargs)
            return y, self.log(logger_state, logs)

        return wrapped

    def tap(
        self,
        f: Callable,
        logger_state: LoggerStateT,
        argnames: Sequence[str] | None = None,
        prefix: str = "",
    ) -> Callable[..., LoggerStateT]:
        def callback(logs: logdict):
            self.callback(logger_state, logs)

        return tap(f, callback=callback, argnames=argnames, prefix=prefix)
