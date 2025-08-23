import lox
from functools import wraps
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Sequence


@dataclass
class LoggerState:
    pass


LoggerStateT = TypeVar("LoggerStateT", bound=LoggerState)


class Logger(Generic[LoggerStateT]):

    def init(self, *args, **kwargs) -> LoggerStateT:
        raise NotImplemented

    def log(self, logger_state: LoggerStateT, logs: lox.logdict) -> None:
        raise NotImplemented

    def spool(
        self, 
        logger_state: LoggerStateT, 
        f: Callable
    ) -> Callable:
        """
        Wraps a function to log its output.

        Args:
          logger_state: The state of the logger.
          f: The function to be wrapped.

        Returns:
          A wrapped function that logs its output.
        """

        @wraps(f)
        def wrapped(*args, **kwargs):
            y, logs = lox.spool(f)(*args, **kwargs)
            self.log(logger_state, logs)
            return y

        return wrapped

    def tap(self, logger_state: LoggerStateT, f: Callable) -> Callable:
        raise NotImplemented


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

    def log(self, logger_state: MultiLoggerState, logs: lox.logdict):
        for sub_logger, sub_logger_state in zip(
            self.loggers, logger_state.logger_states
        ):
            sub_logger.log(sub_logger_state, logs)

    def tap(
        self, 
        logger_state: MultiLoggerState, 
        f: Callable, 
        argnames: Sequence[str] = (),
    ) -> Callable:
        f_tapped = f
        for sub_logger, sub_logger_state in zip(self.loggers, logger_state.logger_states):
            f_tapped = sub_logger.tap(sub_logger_state, f_tapped, argnames=argnames)
        return f_tapped
