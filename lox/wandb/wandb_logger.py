from dataclasses import dataclass

import jax

from lox import logdict
from lox.loggers.logger import Logger, LoggerState
from lox.utils.typing import Key
from lox.wandb.core import WandbRun, finish, init, log


@jax.tree_util.register_dataclass
@dataclass
class WandbLoggerState(LoggerState):
    wandb_run: WandbRun


class WandbLogger(Logger[WandbLoggerState]):
    wandb_kwargs: dict

    def __init__(self, **kwargs):
        super().__init__()
        # wandb's pydantic Settings rejects a non-string id outright, and ids
        # routinely arrive as ints (a SLURM job id resolved through OmegaConf);
        # coerce here so every caller doesn't have to.
        if kwargs.get("id") is not None:
            kwargs["id"] = str(kwargs["id"])
        self.wandb_kwargs = kwargs

    def init(self, key: Key) -> WandbLoggerState:
        wandb_run = init(key, **self.wandb_kwargs)
        return WandbLoggerState(wandb_run=wandb_run)

    def callback(self, state: WandbLoggerState, logs: logdict) -> None:
        log(state.wandb_run, logs)

    def close(self, state: WandbLoggerState) -> None:
        finish(state.wandb_run)
