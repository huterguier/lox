from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Generic, Optional, Sequence, TypeVar

import jax

from lox.logdict import logdict
from lox.spooling import spool


@jax.tree_util.register_dataclass
@dataclass
class LoggerState:
    pass


LoggerStateT = TypeVar("LoggerStateT", bound=LoggerState)


class Logger(Generic[LoggerStateT], ABC):

    @abstractmethod
    def init(self, *args, **kwargs) -> LoggerStateT:
        pass

    @abstractmethod
    def log(self, logger_state: LoggerStateT, logs: logdict, prefix: str = "") -> None:
        pass

    def spool(
        self,
        f: Callable,
        logger_state: LoggerStateT,
        keep_logs: bool = False,
        interval: Optional[int] = None,
        reduce: Optional[str] = None,
        prefix: str = "",
    ) -> Callable:
        """
        Wraps a function to log its output.

        Args:
            logger_state: The state of the logger.
            f: The function to be wrapped.
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
            self.log(logger_state, logs)
            return y

        return wrapped

    @abstractmethod
    def tap(
        self,
        f: Callable,
        logger_state: LoggerStateT,
        argnames: Optional[Sequence[str]] = None,
        prefix: str = "",
    ) -> Callable:
        pass
