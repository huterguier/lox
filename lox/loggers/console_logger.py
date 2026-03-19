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
from lox.tapping import tap


@jax.tree_util.register_dataclass
@dataclass
class ConsoleLoggerState(LoggerState):
    key: jax.Array
    id: jax.Array


class ConsoleLogger(Logger[ConsoleLoggerState]):
    """
    A logger that outputs logs to stdout.
    """

    console: Console
    logss: dict[str, logdict]
    live: Live

    def __init__(self):
        self.console = Console()
        self.logss = {}

    def init(self, key: jax.Array) -> ConsoleLoggerState:
        def callback(key):
            id = jnp.int32(len(self.logss.keys()))
            self.logss[str(id)] = logdict({})
            table = Table(
                box=box.ROUNDED,
                expand=True,
                show_header=False,
                border_style="white",
            )
            self.live = Live(table, console=self.console, refresh_per_second=4)
            self.live.start()
            return id

        id = jax.experimental.io_callback(
            callback,
            jax.ShapeDtypeStruct((), jnp.int32),
            key=key,
        )

        return ConsoleLoggerState(key=key, id=id)

    def callback(self, logger_state: ConsoleLoggerState, logs: logdict):
        id = str(logger_state.id)
        self.logss[id] |= logs
        table = Table(
            box=box.ROUNDED,
            expand=True,
            show_header=False,
            border_style="white",
        )
        try:
            logss = jax.tree.map(lambda *x: jnp.stack(x), *list(self.logss.values()))
        except Exception as e:
            return

        for k, v in logss.items():
            table.add_row(
                f"[bold]{k}[/bold]",
                f"{jnp.mean(v, axis=0)[0]} ± {jnp.std(v, axis=0)[0]}",
            )
        self.live.update(table)
