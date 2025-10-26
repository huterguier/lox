from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jax

from lox import logdict, tap
from lox.loggers.logger import Logger, LoggerState
from lox.wandb.core import WandbRun, init, log


@jax.tree_util.register_dataclass
@dataclass
class WandbLoggerState(LoggerState):
    wandb_run: WandbRun


class WandbLogger(Logger[WandbLoggerState]):
    wandb_kwargs: dict

    def __init__(self, **kwargs):
        super().__init__()
        self.wandb_kwargs = kwargs

    def init(self, key: jax.Array) -> WandbLoggerState:
        wandb_run = init(key, **self.wandb_kwargs)
        return WandbLoggerState(wandb_run=wandb_run)

    def log(self, logger_state: WandbLoggerState, logs: logdict):
        log(logger_state.wandb_run, logs)

    def tap(
        self,
        f: Callable,
        logger_state: WandbLoggerState,
        argnames: Optional[Sequence[str]] = None,
        prefix: str = "",
    ) -> Callable:
        def callback(logs: logdict):
            log(logger_state.wandb_run, logs)

        return tap(f, callback=callback, argnames=argnames, prefix=prefix)
