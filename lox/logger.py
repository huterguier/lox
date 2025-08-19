import lox
from functools import wraps
from dataclasses import dataclass
from typing import Callable


@dataclass
class LoggerState:
    pass


class Logger:

  def init(self, *args, **kwargs) -> LoggerState:
    raise NotImplemented

  def log(self, logger_state: LoggerState, logs: lox.logdict):
    raise NotImplemented

  def spool(self, logger_state: LoggerState, f: Callable) -> Callable:
    raise NotImplemented

  def tap(self, logger_state: LoggerState, f: Callable) -> Callable:
    raise NotImplemented

  def close(self, logger_state: LoggerState):
    raise NotImplemented


@dataclass
class MultiLoggerState(LoggerState):
  logger_states: list[LoggerState]


class MultiLogger(Logger):
  loggers: list

  def __init__(self, loggers):
    self.loggers = loggers

  def init(self, *args, **kwargs) -> MultiLoggerState:
    logger_states = [logger.init(*args, **kwargs) for logger in self.loggers]
    return MultiLoggerState(logger_states=logger_states)

  def log(self, logger_state: LoggerState, logs: lox.logdict):
    for logger in self.loggers:
      logger.log(logger_state, logs)

  def spool(self, logger_state: MultiLoggerState, f: Callable) -> Callable:
    @wraps(f)
    def wrapped(*args, **kwargs):
      y, logs = lox.spool(f)(*args, **kwargs)
      for logger in self.loggers:
        logger.log(logger_state, logs)
      return y
    return wrapped

  def tap(self, logger_state: LoggerState, f: Callable) -> Callable:
    @wraps(f)
    def wrapped(*args, **kwargs):
      f_tapped = f
      for logger in self.loggers:
        f_tapped = logger.tap(logger_state, f_tapped)
      y = f_tapped(*args, **kwargs)
      return y

  def close(self, logger_state):
    raise NotImplemented
