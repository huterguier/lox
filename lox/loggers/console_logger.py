import atexit
import math
import threading
from dataclasses import dataclass

import jax
import jax.experimental
import jax.numpy as jnp
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import ProgressBar
from rich.table import Table
from rich.text import Text

from lox.logdict import logdict
from lox.loggers.logger import Logger, LoggerState


@jax.tree_util.register_dataclass
@dataclass
class ConsoleLoggerState(LoggerState):
    key: jax.Array
    id: jax.Array


def _sections(keys: list[str]) -> dict[str, list[str]]:
    """Groups keys by the part before their first ``/``.

    Keys without a ``/`` share the leading unnamed section, so they stay at the top
    rather than being scattered between the named ones.
    """
    sections: dict[str, list[str]] = {"": []}
    for k in keys:
        section = k.split("/")[0] if "/" in k else ""
        sections.setdefault(section, []).append(k)
    if not sections[""]:
        del sections[""]
    return sections


_VALUE_ALLOWANCE = 16  # widest realistic "mean ± std"
_GUTTER = 3
_MAX_COLUMNS = 4


def _runs(n: int) -> str:
    return f"{n} run" + ("s" if n != 1 else "")


def _number(value: float) -> str:
    """Formats a statistic for display.

    Large magnitudes use thousands separators rather than scientific notation, so
    that step counters stay readable; everything else keeps four significant
    digits, which suits both ordinary metrics and very small ones.
    """
    return f"{value:,.0f}" if abs(value) >= 1e4 else f"{value:.4g}"


def _shape(values: list) -> str:
    """Names the shape each run contributed, which differs from key to key."""
    shapes = {value.shape for value in values}
    return "mixed shapes" if len(shapes) > 1 else str(shapes.pop())


def _shape_and_runs(values: list) -> str:
    return f"{_shape(values)} · {_runs(len(values))}"


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

    def __init__(self, progress: dict[str, float] | None = None):
        """
        Args:
            progress: Maps a logged key to the total it counts towards, rendering
                it as a progress bar above the table instead of as a row. Bars only
                advance during a run under :meth:`tap`; under :meth:`spool` the logs
                arrive in one callback once the function has returned.
        """
        self.console = Console()
        self.logss = {}
        self.live = None
        self.progress = dict(progress or {})
        self._lock = threading.Lock()

    def init(self, key: jax.Array) -> ConsoleLoggerState:
        def callback(key):
            with self._lock:
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
        # The spacer column takes all the slack, so a value stays beside its name
        # while the shape lines up with the bar's counter at the right edge.
        table = Table(box=None, expand=True, show_header=False, pad_edge=False)
        table.add_column(no_wrap=True)
        table.add_column(justify="right", no_wrap=True, style="bold")
        table.add_column(ratio=1)
        table.add_column(justify="right", style="dim", no_wrap=True)
        return table

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
        # tap routes logs through an unordered jax.debug.callback, so two lanes can
        # arrive at once. Without the lock one could insert a run while the other
        # is iterating logss to build the table.
        with self._lock:
            self._render(logger_state, logs)

    def _render(self, logger_state: ConsoleLoggerState, logs: logdict):
        id = str(logger_state.id)
        self.logss.setdefault(id, logdict({}))
        self.logss[id] |= logdict(_flatten(logs))

        self._start()
        keys = {k for run in self.logss.values() for k in run} - set(self.progress)
        sections = _sections(sorted(keys))

        # The run count is normally the same on every row, so it belongs in the
        # subtitle rather than repeated down a column. Shapes differ per key and
        # stay on the row. When the counts disagree -- while runs are still
        # reporting, or for a key only some of them log -- there is no single
        # count to state, so each row carries its own instead.
        counts = {len([run for run in self.logss.values() if k in run]) for k in keys}
        rendered = []
        for section, section_keys in sections.items():
            rows = []
            for k in section_keys:
                values = [run[k] for run in self.logss.values() if k in run]
                # A complex value has no mean the terminal can show, so summarise
                # its magnitude rather than raising from float() further down.
                values = [
                    jnp.abs(value) if jnp.iscomplexobj(value) else value
                    for value in values
                ]
                if len(values) > 1:
                    # Runs are separate dict entries rather than an array axis, so
                    # the spread across them is recoverable: reduce each run first,
                    # then report how much the runs disagree.
                    v = jnp.stack([jnp.mean(jnp.ravel(value)) for value in values])
                else:
                    v = jnp.ravel(values[0])
                summary = _number(float(jnp.mean(v)))
                if v.size > 1:
                    summary += f" ± {_number(float(jnp.std(v)))}"
                label = k.removeprefix(f"{section}/") if section else k
                rows.append(
                    (
                        f"{'  ' if section else ''}{label}",
                        summary,
                        _shape(values) if len(counts) == 1 else _shape_and_runs(values),
                    )
                )
            rendered.append((section, rows))

        grid = self._grid(rendered)
        bars = self._bars()
        self.live.update(
            Panel(
                Group(grid, Text(""), bars) if bars.row_count else grid,
                box=box.ROUNDED,
                border_style="dim",
                subtitle=_runs(counts.pop()) if len(counts) == 1 else None,
                subtitle_align="right",
            )
        )

    def _columns(self, sections: list) -> int:
        """Chooses how many columns the metrics are laid out in.

        The estimate deliberately uses only widths that do not move as values do
        -- key names, shapes, and a fixed allowance for the number -- so that a
        metric growing from ``5`` to ``1,234,567`` cannot make the whole layout
        flip between column counts on successive refreshes.
        """
        rows = [row for _, section_rows in sections for row in section_rows]
        if not rows:
            return 1
        width = (
            max(len(label) for label, _, _ in rows)
            + _VALUE_ALLOWANCE
            + max(len(detail) for _, _, detail in rows)
            + 4  # inter-column padding within a sub-table
        )
        available = self.console.width - 4  # panel border and padding
        fits = available // (width + _GUTTER)
        return max(1, min(_MAX_COLUMNS, len(sections), fits))

    def _grid(self, sections: list) -> Table:
        """Lays the sections out side by side, in order, column by column."""
        n = self._columns(sections)
        grid = Table.grid(expand=True, padding=(0, _GUTTER))
        for _ in range(n):
            grid.add_column(ratio=1)

        heights = [len(rows) + (2 if section else 0) for section, rows in sections]
        target = max(1, math.ceil(sum(heights) / n))
        columns: list[list] = []
        current: list = []
        used = 0
        for (section, rows), height in zip(sections, heights, strict=True):
            if current and used + height > target and len(columns) < n - 1:
                columns.append(current)
                current, used = [], 0
            current.append((section, rows))
            used += height
        columns.append(current)

        tables = []
        for column in columns:
            table = self._new_table()
            for i, (section, rows) in enumerate(column):
                if section:
                    if i:
                        table.add_row("", "", "", "")
                    table.add_row(f"[bold cyan]{section}[/bold cyan]", "", "", "")
                for label, summary, detail in rows:
                    table.add_row(label, summary, "", detail)
            tables.append(table)
        grid.add_row(*tables, *[""] * (n - len(tables)))
        return grid

    def _bars(self) -> Table:
        """Renders one bar per configured key, at the mean progress of the runs.

        Bars live in their own table so the metric columns are not stretched to
        accommodate a full-width bar.
        """
        table = Table(box=None, expand=True, show_header=False, pad_edge=False)
        table.add_column(no_wrap=True)
        table.add_column(ratio=1)
        table.add_column(justify="right", style="dim", no_wrap=True)
        for k, total in self.progress.items():
            values = [run[k] for run in self.logss.values() if k in run]
            if not values:
                continue
            # max() rather than the last element: the leading axis has no reliable
            # order, and a counter only ever grows. Under vmap the lanes advance in
            # lockstep, so their mean is simply the shared position.
            completed = float(jnp.mean(jnp.stack([jnp.max(value) for value in values])))
            table.add_row(
                k,
                ProgressBar(
                    total=total,
                    completed=min(completed, total),
                    complete_style="cyan",
                    finished_style="green",
                ),
                f"{completed:,.0f}/{total:,.0f}",
            )
        return table
