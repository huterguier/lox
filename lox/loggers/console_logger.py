import atexit
import math
import threading
from dataclasses import dataclass
from typing import NamedTuple

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

_VALUE_ALLOWANCE = 16
_GUTTER = 3
_MAX_COLUMNS = 4
_CELL_PADDING = 3
_PANEL_CHROME = 4


class Row(NamedTuple):
    """One rendered metric: its name, what is shown beside it, and its statistic."""

    label: str
    detail: str
    summary: str


Section = tuple[str, list[Row]]
"""A section name -- empty for ungrouped keys -- and the rows beneath it."""


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
    for key in keys:
        section = key.split("/")[0] if "/" in key else ""
        sections.setdefault(section, []).append(key)
    if not sections[""]:
        del sections[""]
    return sections


def _flatten(data: dict, prefix: str = "") -> dict:
    """Flattens nested log dicts into ``"outer/inner"`` keys."""
    flat = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}/"))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _runs(n: int) -> str:
    return f"{n} run" + ("s" if n != 1 else "")


def _number(value: float) -> str:
    """Formats a statistic for display.

    Large magnitudes use thousands separators rather than scientific notation, so
    that step counters stay readable; everything else keeps four significant
    digits, which suits both ordinary metrics and very small ones.
    """
    return f"{value:,.0f}" if abs(value) >= 1e4 else f"{value:.4g}"


def _shape(values: list[jax.Array]) -> str:
    """Names the shape each run contributed, which differs from key to key.

    A leading axis of size 1 is dropped. ``lox.log`` gives every value such an
    axis for the log event, and it is the callback's per-lane dispatch -- not the
    array -- that carries ``vmap``, so the event axis stays leading. Dropping only
    that one leaves any size-1 axis of the logged value itself intact.
    """
    shapes = {
        value.shape[1:] if value.shape[:1] == (1,) else value.shape for value in values
    }
    return "mixed shapes" if len(shapes) > 1 else str(shapes.pop())


def _detail(values: list[jax.Array], show_runs: bool) -> str:
    """Describes a row beyond its name, or "" when there is nothing to add.

    A scalar's shape says nothing the value does not already say, so it is left
    out; anything else -- a vector, an image, differing shapes across runs -- is
    named, as is the run count when rows disagree about it.
    """
    parts = []
    shape = _shape(values)
    if shape != "()":
        parts.append(shape)
    if show_runs:
        parts.append(_runs(len(values)))
    return " · ".join(parts)


def _summarise(values: list[jax.Array]) -> str:
    """Reduces a key's values across runs to a ``mean ± std`` cell.

    Runs are separate dict entries rather than an array axis, so the spread across
    them is recoverable: with more than one, each run is reduced first and the
    deviation then reports how much the runs disagree.
    """
    if len(values) > 1:
        reduced = jnp.stack([jnp.mean(jnp.ravel(value)) for value in values])
    else:
        reduced = jnp.ravel(values[0])
    summary = _number(float(jnp.mean(reduced)))
    if reduced.size > 1:
        summary += f" ± {_number(float(jnp.std(reduced)))}"
    return summary


class ConsoleLogger(Logger[ConsoleLoggerState]):
    """
    A logger that renders logs as a live-updating panel on stdout.

    Each call to :meth:`init` registers a new run, and the panel shows one row per
    logged key with a mean and a standard deviation. Keys are grouped into sections
    by the part before their first ``/``, and the sections are laid out in several
    columns when the terminal is wide enough.

    With a single run, both statistics are taken over every value logged under that
    key. With several runs, each run is reduced first and the deviation is taken
    across the per-run means, so it measures how much the runs disagree rather than
    how much a metric moves within a run. Getting that across-run number requires
    one :meth:`init` per run -- either in a Python loop or under ``vmap`` -- since
    runs sharing a single state are indistinguishable once logged.

    Within a run the reduction is deliberately order-independent. The leading axis
    of a logged value mixes scan iterations, ``vmap`` lanes and separate ``lox.log``
    call sites together, so no element of it can be identified as "the latest" --
    only aggregate statistics are meaningful. The shape each run contributed is
    named beside the row instead, so the discarded structure stays visible.

    Each call replaces the values of the keys it logs, so the panel reflects the
    most recent call rather than the whole session.
    """

    console: Console
    logss: dict[str, logdict]
    live: Live | None

    def __init__(self, progress: dict[str, float] | None = None):
        """
        Args:
            progress: Maps a logged key to the total it counts towards, rendering
                it as a progress bar below the metrics instead of as a row. Bars
                only advance during a run under :meth:`tap`; under :meth:`spool`
                the logs arrive in one callback once the function has returned.
        """
        self.console = Console()
        self.logss = {}
        self.live = None
        self.progress = dict(progress or {})
        self._lock = threading.Lock()

    def init(self, key: jax.Array) -> ConsoleLoggerState:
        def callback(key):
            with self._lock:
                run_id = jnp.int32(len(self.logss.keys()))
                self.logss[str(run_id)] = logdict({})
                self._start()
            return run_id

        run_id = jax.experimental.io_callback(
            callback,
            jax.ShapeDtypeStruct((), jnp.int32),
            key=key,
        )

        return ConsoleLoggerState(key=key, id=run_id)

    def _start(self) -> None:
        """Starts the live display, reusing it across runs."""
        if self.live is None:
            self.live = Live(Text(""), console=self.console, refresh_per_second=4)
            self.live.start()
            atexit.register(self.close)

    def close(self) -> None:
        """Stops the live display and restores the terminal."""
        if self.live is not None:
            self.live.stop()
            self.live = None

    def callback(self, logger_state: ConsoleLoggerState, logs: logdict):
        """Records the logs and redraws the panel.

        Rendering is serialised: ``tap`` routes logs through an *unordered*
        ``jax.debug.callback``, so two lanes can arrive at once and one could
        register a run while the other iterates them to build the panel.
        """
        with self._lock:
            self._render(logger_state, logs)

    def _values(self, key: str) -> list[jax.Array]:
        """Each run's latest array for a key, skipping runs that lack it.

        Complex values are reduced to their magnitude: they have no mean the
        terminal can show, and leaving them would raise from ``float()`` later.
        """
        return [
            jnp.abs(run[key]) if jnp.iscomplexobj(run[key]) else run[key]
            for run in self.logss.values()
            if key in run
        ]

    def _render(self, logger_state: ConsoleLoggerState, logs: logdict):
        """Rebuilds the panel from every run recorded so far.

        The run count is normally the same on every row, so it is stated once in
        the subtitle rather than repeated down a column. When the counts disagree
        -- while runs are still reporting, or for a key only some of them log --
        there is no single count to state and each row carries its own instead.
        """
        run_id = str(logger_state.id)
        self.logss.setdefault(run_id, logdict({}))
        self.logss[run_id] |= logdict(_flatten(logs))
        self._start()

        keys = {k for run in self.logss.values() for k in run} - set(self.progress)
        counts = {len(self._values(k)) for k in keys}
        show_runs = len(counts) > 1

        sections: list[Section] = []
        for section, section_keys in _sections(sorted(keys)).items():
            rows = []
            for key in section_keys:
                values = self._values(key)
                label = key.removeprefix(f"{section}/") if section else key
                rows.append(
                    Row(
                        label=f"{'  ' if section else ''}{label}",
                        detail=_detail(values, show_runs=show_runs),
                        summary=_summarise(values),
                    )
                )
            sections.append((section, rows))

        grid = self._grid(sections)
        bars = self._bars()
        self.live.update(
            Panel(
                Group(grid, Text(""), bars) if bars.row_count else grid,
                box=box.ROUNDED,
                border_style="dim",
                subtitle=_runs(next(iter(counts))) if len(counts) == 1 else None,
                subtitle_align="right",
            )
        )

    def _new_table(self) -> Table:
        """Builds an empty table of name, detail, value and a trailing spacer.

        The detail sits immediately behind the name rather than in a far column,
        but keeps a cell of its own so values stay aligned across rows. The spacer
        takes the slack, so a value stays beside its name instead of drifting to
        the far edge.
        """
        table = Table(box=None, expand=True, show_header=False, pad_edge=False)
        table.add_column(no_wrap=True)
        table.add_column(no_wrap=True, style="dim")
        table.add_column(justify="right", no_wrap=True, style="bold")
        table.add_column(ratio=1)
        return table

    def _columns(self, sections: list[Section]) -> int:
        """Chooses how many columns the sections are laid out in.

        The estimate deliberately uses only widths that do not move as values do
        -- key names, shapes, and a fixed allowance for the number -- so that a
        metric growing from ``5`` to ``1,234,567`` cannot make the whole layout
        flip between column counts on successive refreshes. ``_VALUE_ALLOWANCE``
        is that allowance, chosen as the widest realistic ``mean ± std``, and the
        two constants beside it cover the cell padding and the panel border.
        """
        rows = [row for _, section_rows in sections for row in section_rows]
        if not rows:
            return 1
        width = (
            max(len(label) for label, _, _ in rows)
            + max(len(detail) for _, detail, _ in rows)
            + _VALUE_ALLOWANCE
            + _CELL_PADDING
        )
        available = self.console.width - _PANEL_CHROME
        return max(1, min(_MAX_COLUMNS, len(sections), available // (width + _GUTTER)))

    def _pack(self, sections: list[Section], n: int) -> list[list[Section]]:
        """Splits the sections into ``n`` columns, in order.

        Filling column by column keeps sections where they would be in a single
        column, at the cost of columns that can differ in height.
        """
        heights = [len(rows) + (2 if section else 0) for section, rows in sections]
        target = max(1, math.ceil(sum(heights) / n))
        columns: list[list[Section]] = []
        current: list[Section] = []
        used = 0
        for section, height in zip(sections, heights, strict=True):
            if current and used + height > target and len(columns) < n - 1:
                columns.append(current)
                current, used = [], 0
            current.append(section)
            used += height
        columns.append(current)
        return columns

    def _grid(self, sections: list[Section]) -> Table:
        """Lays the sections out side by side."""
        n = self._columns(sections)
        grid = Table.grid(expand=True, padding=(0, _GUTTER))
        for _ in range(n):
            grid.add_column(ratio=1)

        tables = []
        for column in self._pack(sections, n):
            table = self._new_table()
            for i, (section, rows) in enumerate(column):
                if section:
                    if i:
                        table.add_row("", "", "", "")
                    table.add_row(f"[bold cyan]{section}[/bold cyan]", "", "", "")
                for label, detail, summary in rows:
                    table.add_row(label, detail, summary, "")
            tables.append(table)
        grid.add_row(*tables, *[""] * (n - len(tables)))
        return grid

    def _bars(self) -> Table:
        """Renders one bar per configured key, at the mean progress of the runs.

        Progress is the maximum logged value rather than the last: the leading
        axis has no reliable order, and a counter only ever grows. Under ``vmap``
        the lanes advance in lockstep, so their mean is simply the shared position.

        Bars live in their own table so the metric columns are not stretched to
        accommodate a full-width bar.
        """
        table = Table(box=None, expand=True, show_header=False, pad_edge=False)
        table.add_column(no_wrap=True)
        table.add_column(ratio=1)
        table.add_column(justify="right", style="dim", no_wrap=True)
        for key, total in self.progress.items():
            values = self._values(key)
            if not values:
                continue
            completed = float(jnp.mean(jnp.stack([jnp.max(value) for value in values])))
            table.add_row(
                key,
                ProgressBar(
                    total=total,
                    completed=min(completed, total),
                    complete_style="cyan",
                    finished_style="green",
                ),
                f"{completed:,.0f}/{total:,.0f}",
            )
        return table
