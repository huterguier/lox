from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jax
import jax.experimental
import jax.numpy as jnp
from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table

from lox.logdict import logdict
from lox.loggers.logger import Logger, LoggerState
from lox.tap import tap


@jax.tree_util.register_dataclass
@dataclass
class ConsoleLoggerState(LoggerState):
    key: jax.Array
    id: jax.Array


def make_dashboard(logs: logdict) -> Table:
    table = Table(
        box=box.ROUNDED,
        expand=True,
        show_header=False,
        border_style="white",
    )
    for k, v in logs.items():
        table.add_row(f"[bold]{k}[/bold]", f"{v[-1]:.4f}")
    return table


class ConsoleLogger(Logger[ConsoleLoggerState]):
    """
    A logger that outputs logs to stdout.
    """

    console: Console
    logss: dict[str, logdict]
    lives: dict[str, Live]

    def __init__(self):
        self.console = Console()
        self.logss = {}
        self.lives = {}

    def init(self, key: jax.Array) -> ConsoleLoggerState:
        def callback(key):
            id = jnp.int32(len(self.lives))
            self.logss[str(id)] = logdict({})
            table = Table(
                box=box.ROUNDED,
                expand=True,
                show_header=False,
                border_style="white",
            )
            self.lives[str(id)] = Live(
                table, console=self.console, refresh_per_second=4
            )
            self.lives[str(id)].start()
            return id

        id = jax.experimental.io_callback(
            callback,
            jax.ShapeDtypeStruct((), jnp.int32),
            key=key,
        )

        return ConsoleLoggerState(key=key, id=id)

    def log(self, logger_state: ConsoleLoggerState, logs: logdict) -> None:
        def callback(logger_state, logs):
            id = str(logger_state.id)
            self.logss[id] += logs
            table = make_dashboard(logs)
            self.lives[id].update(table)

        jax.debug.callback(
            callback,
            logger_state=logger_state,
            logs=logs,
        )

    def tap(
        self,
        f: Callable,
        logger_state: ConsoleLoggerState,
        argnames: Optional[Sequence[str]] = None,
    ) -> Callable:
        def callback(logs):
            id = str(logger_state.id)
            self.logss[id] += logs
            table = make_dashboard(self.logss[id])
            self.lives[id].update(table)

        return tap(f, callback=callback, argnames=argnames)
