import jax
import lox
from dataclasses import dataclass
from typing import Callable, Optional
from lox.logger import Logger, LoggerState


@dataclass
class WandbLoggerState(LoggerState):
    run: lox.wandb.Run


class WandbLogger(Logger[WandbLoggerState]):
    wandb_kwargs: dict

    def __init__(self, **kwargs):
        super().__init__()
        self.wandb_kwargs = kwargs

    def init(self, key: jax.Array) -> WandbLoggerState:
        run = lox.wandb.init(key, **self.wandb_kwargs)
        return WandbLoggerState(run=run)

    def log(self, logger_state: WandbLoggerState, logs: lox.logdict):
        run = logger_state.run
        lox.wandb.log(run, logs)

    def tap(self, logger_state: WandbLoggerState, f: Callable) -> Callable:
        def callback(logs: lox.logdict):
            run = logger_state.run
            lox.wandb.log(run, logs)

        return lox.tap(f, callback=callback)
