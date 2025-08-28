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
        if "step" in logs.steps:
            for k, vs in logs.items():
                steps = logs.steps["step"]
                for v, step in zip(vs, steps):
                    lox.wandb.log(run, {k: v})
        else:
            leaves, structure = jax.tree.flatten(logs)
            assert all([len(leaf) == len(leaves[0]) for leaf in leaves])
            for i in range(len(leaves[0])):
                lox.wandb.log(
                    run, jax.tree.unflatten(structure, [leaf[i] for leaf in leaves])
                )

    def tap(self, logger_state: WandbLoggerState, f: Callable) -> Callable:
        def callback(logs: lox.logdict):
            run = logger_state.run
            lox.wandb.log(run, logs)

        return lox.tap(f, callback=callback)
