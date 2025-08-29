import jax
from dataclasses import dataclass
from typing import Callable, Optional, Sequence
from lox.logger import Logger, LoggerState
from lox import logdict, tap
from .run import WandbRun
from .init import init, log


@dataclass
class WandbLoggerState(LoggerState):
    run: WandbRun


class WandbLogger(Logger[WandbLoggerState]):
    wandb_kwargs: dict

    def __init__(self, **kwargs):
        super().__init__()
        self.wandb_kwargs = kwargs

    def init(self, key: jax.Array) -> WandbLoggerState:
        run = init(key, **self.wandb_kwargs)
        return WandbLoggerState(run=run)

    def log(self, logger_state: WandbLoggerState, logs: logdict):
        run = logger_state.run
        log(run, logs)

    def tap(
        self,
        f: Callable,
        logger_state: WandbLoggerState,
        argnames: Optional[Sequence[str]] = None,
    ) -> Callable:
        def callback(logs: logdict):
            run = logger_state.run
            log(run, logs)

        return tap(f, callback=callback, argnames=argnames)
