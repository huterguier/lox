import atexit
from dataclasses import dataclass

import jax
import jax.experimental
import jax.numpy as jnp
from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table

from lox.logdict import logdict
from lox.loggers.logger import Logger, LoggerState


@jax.tree_util.register_dataclass
@dataclass
class ConsoleLoggerState(LoggerState):
    key: jax.Array
    id: jax.Array


def _flatten(data: dict, prefix: str = "") -> dict:
    """Flattens nested log dicts into ``"outer/inner"`` keys."""
    flat = {}
    for k, v in data.items():
        if isinstance(v, dict):
            flat.update(_flatten(v, f"{prefix}{k}/"))
        else:
            flat[f"{prefix}{k}"] = v
    return flat


class ConsoleLogger(Logger[ConsoleLoggerState]):
    """
    A logger that renders logs as a live-updating table on stdout.

    Each call to :meth:`init` registers a new run, and the table shows one row per
    logged key with a mean and a standard deviation.

    With a single run, both are taken over every value logged under that key. With
    several runs, each run is reduced first and the deviation is taken across the
    per-run means, so it measures how much the runs disagree rather than how much a
    metric moves within a run. Getting that across-run number requires one
    :meth:`init` per run -- either in a Python loop or under ``vmap`` -- since runs
    sharing a single state are indistinguishable once logged.

    Within a run the reduction is deliberately order-independent. The leading axis
    of a logged value mixes scan iterations, ``vmap`` lanes and separate ``lox.log``
    call sites together, so no element of it can be identified as "the latest" --
    only aggregate statistics are meaningful. The shape each run contributed is
    reported alongside the row instead.

    Each call replaces the values of the keys it logs, so the table reflects the
    most recent call rather than the whole session.
    """

    console: Console
    logss: dict[str, logdict]
    live: Live | None

    def __init__(self):
        self.console = Console()
        self.logss = {}
        self.live = None

    def init(self, key: jax.Array) -> ConsoleLoggerState:
        def callback(key):
            id = jnp.int32(len(self.logss.keys()))
            self.logss[str(id)] = logdict({})
            self._start()
            return id

        id = jax.experimental.io_callback(
            callback,
            jax.ShapeDtypeStruct((), jnp.int32),
            key=key,
        )

        return ConsoleLoggerState(key=key, id=id)

    def _new_table(self) -> Table:
        return Table(
            box=box.ROUNDED,
            expand=True,
            show_header=False,
            border_style="white",
        )

    def _start(self) -> None:
        """Starts the live display, reusing it across runs."""
        if self.live is None:
            self.live = Live(
                self._new_table(), console=self.console, refresh_per_second=4
            )
            self.live.start()
            atexit.register(self.close)

    def close(self) -> None:
        """Stops the live display and restores the terminal."""
        if self.live is not None:
            self.live.stop()
            self.live = None

    def callback(self, logger_state: ConsoleLoggerState, logs: logdict):
        id = str(logger_state.id)
        self.logss.setdefault(id, logdict({}))
        self.logss[id] |= logdict(_flatten(logs))

        self._start()
        table = self._new_table()
        for k in sorted({k for run in self.logss.values() for k in run}):
            values = [run[k] for run in self.logss.values() if k in run]
            if len(values) > 1:
                # Runs are separate dict entries rather than an array axis, so the
                # spread across them is recoverable: reduce each run first, then
                # report how much the runs disagree.
                v = jnp.stack([jnp.mean(jnp.ravel(value)) for value in values])
            else:
                v = jnp.ravel(values[0])
            table.add_row(
                f"[bold]{k}[/bold]",
                f"{float(jnp.mean(v)):.4g} ± {float(jnp.std(v)):.4g}",
                f"[dim]{self._detail(values)}[/dim]",
            )
        self.live.update(table)

    def _detail(self, values: list[jax.Array]) -> str:
        """Describes which runs a row covers and what each contributed."""
        n_runs = len(self.logss)
        if len(values) < n_runs:
            runs = f"{len(values)}/{n_runs} runs"
        else:
            runs = f"{n_runs} run" + ("s" if n_runs != 1 else "")

        shapes = {value.shape for value in values}
        if len(shapes) > 1:
            return f"{runs}, mixed shapes"
        shape = shapes.pop()
        return f"{runs}, {'each ' if len(values) > 1 else ''}{shape}"
