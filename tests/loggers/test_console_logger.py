import threading
import time

import jax
import jax.numpy as jnp
import pytest
from rich.console import Group
from rich.text import Text
from test_logger import TestLogger

import lox
from lox.logdict import logdict
from lox.loggers import ConsoleLogger


class TestConsoleLogger(TestLogger):
    @pytest.fixture
    def logger(self):
        console_logger = ConsoleLogger()
        yield console_logger
        console_logger.close()


def table(logger):
    """Returns the metrics table from inside the panel, ahead of any bars."""
    renderable = logger.live.get_renderable().renderable
    return renderable.renderables[0] if isinstance(renderable, Group) else renderable


def subtitle(logger) -> str | None:
    return logger.live.get_renderable().subtitle


def _plain(cells) -> list[str]:
    """Strips styling so assertions do not depend on the palette."""
    return [Text.from_markup(cell).plain for cell in cells]


def names(logger) -> list[str]:
    """Lists the first column verbatim, including blank spacer rows."""
    return _plain(table(logger).columns[0]._cells)


def rendered(logger) -> dict[str, str]:
    """Maps each rendered row's key to its ``mean ± std`` cell."""
    columns = table(logger).columns
    return dict(zip(_plain(columns[0]._cells), _plain(columns[1]._cells), strict=True))


def details(logger) -> dict[str, str]:
    """Maps each rendered row's key to its trailing detail cell."""
    columns = table(logger).columns
    return dict(zip(_plain(columns[0]._cells), _plain(columns[-1]._cells), strict=True))


def test_divergent_keys_are_all_rendered():
    logger = ConsoleLogger()
    for seed, logs in enumerate(
        [{"loss": jnp.array([5.0])}, {"accuracy": jnp.array([5.0])}]
    ):
        state = logger.init(jax.random.key(seed))
        logger.callback(state, logdict(logs))
    assert set(rendered(logger)) == {"loss", "accuracy"}
    logger.close()


def test_key_is_pooled_over_runs_that_logged_it():
    logger = ConsoleLogger()
    for seed, value in enumerate([5.0, 7.0]):
        state = logger.init(jax.random.key(seed))
        logger.callback(state, logdict({"loss": jnp.array([value])}))
    assert rendered(logger)["loss"] == "6 ± 1"
    logger.close()


def test_reduction_is_order_independent():
    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.callback(state, logdict({"v": jnp.array([1.0, 2.0, 3.0, 4.0])}))
    ascending = rendered(logger)
    logger.callback(state, logdict({"v": jnp.array([4.0, 3.0, 2.0, 1.0])}))
    assert rendered(logger) == ascending
    logger.close()


def test_nested_logs_are_flattened():
    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.callback(state, logdict({"m": {"a": jnp.ones(3)}}))
    assert set(logger.logss["0"]) == {"m/a"}
    assert names(logger) == ["m", "  a"]
    logger.close()


def test_callback_without_init_does_not_raise():
    other = ConsoleLogger()
    state = other.init(jax.random.key(0))
    other.close()

    logger = ConsoleLogger()
    logger.callback(state, logdict({"x": jnp.ones(2)}))
    assert set(rendered(logger)) == {"x"}
    logger.close()


def test_deviation_across_runs_ignores_within_run_variation():
    # Each run falls 10 -> 1, but the runs barely differ from one another. The
    # reported deviation must describe the disagreement between runs (~0.04), not
    # the decline within them (~3.3).
    logger = ConsoleLogger()
    for seed in range(3):
        state = logger.init(jax.random.key(seed))
        values = jnp.array([10.0, 7.0, 4.0, 2.0, 1.0]) + seed * 0.05
        logger.callback(state, logdict({"loss": values}))
    assert rendered(logger)["loss"] == "4.85 ± 0.04082"
    logger.close()


def test_single_run_deviation_covers_all_values():
    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.callback(state, logdict({"loss": jnp.array([10.0, 7.0, 4.0, 2.0, 1.0])}))
    assert rendered(logger)["loss"] == "4.8 ± 3.311"
    logger.close()


def test_deviation_is_omitted_for_a_lone_value():
    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.callback(state, logdict({"loss": jnp.array([5.0])}))
    assert rendered(logger)["loss"] == "5"

    # A second value to compare against brings the deviation back, even when the
    # values agree and it is therefore zero.
    state = logger.init(jax.random.key(1))
    logger.callback(state, logdict({"loss": jnp.array([5.0])}))
    assert rendered(logger)["loss"] == "5 ± 0"
    logger.close()


def test_deviation_is_kept_for_a_single_run_with_several_values():
    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.callback(state, logdict({"loss": jnp.array([4.0, 6.0])}))
    assert rendered(logger)["loss"] == "5 ± 1"
    logger.close()


def test_shape_is_per_row_and_run_count_is_in_the_subtitle():
    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.callback(state, logdict({"img": jnp.ones((5, 3, 4))}))
    assert details(logger)["img"] == "(5, 3, 4)"
    assert subtitle(logger) == "1 run"

    state = logger.init(jax.random.key(1))
    logger.callback(state, logdict({"img": jnp.ones((5, 3, 4))}))
    assert details(logger)["img"] == "(5, 3, 4)"
    assert subtitle(logger) == "2 runs"
    logger.close()


def test_shapes_differ_between_keys_in_one_run():
    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.callback(state, logdict({"loss": jnp.ones(5), "hist": jnp.ones((5, 4))}))
    assert details(logger) == {
        "hist": "(5, 4)",
        "loss": "(5,)",
    }
    assert subtitle(logger) == "1 run"
    logger.close()


def test_detail_reports_partial_and_mixed_shapes():
    logger = ConsoleLogger()
    for seed, shape in enumerate([(5,), (3,), (5,)]):
        state = logger.init(jax.random.key(seed))
        logs = {"loss": jnp.ones(shape)}
        if seed == 0:
            logs["only_first"] = jnp.ones(2)
        logger.callback(state, logdict(logs))
    # The counts disagree, so no single count can be stated in the subtitle and
    # each row carries its own instead.
    assert details(logger)["loss"] == "mixed shapes · 3 runs"
    assert details(logger)["only_first"] == "(2,) · 1 run"
    assert subtitle(logger) is None
    logger.close()


def test_vmapped_init_yields_one_run_per_lane():
    def f(x):
        lox.log({"v": x.sum()})
        return x

    logger = ConsoleLogger()
    keys = jnp.stack([jax.random.key(seed) for seed in range(3)])
    states = jax.vmap(logger.init)(keys)
    jax.vmap(lambda state, x: logger.spool(f, state)(x))(states, jnp.ones((3, 5)))
    assert details(logger)["v"] == "(1,)"
    assert subtitle(logger) == "3 runs"
    logger.close()


def test_shared_state_across_vmap_lanes_is_a_single_run():
    def f(x):
        lox.log({"v": x.sum()})
        return x

    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.spool(jax.vmap(f), state)(jnp.ones((3, 5)))
    assert details(logger)["v"] == "(3, 1)"
    assert subtitle(logger) == "1 run"
    logger.close()


def test_keys_are_grouped_into_sections():
    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.callback(
        state,
        logdict(
            {
                "lr": jnp.ones(1),
                "train": {"loss": jnp.ones(1)},
                "eval": {"acc": jnp.ones(1)},
            }
        ),
    )
    # Ungrouped keys lead, then one header per section with its members indented
    # and a blank row separating each section from the previous one.
    assert names(logger) == [
        "lr",
        "",
        "eval",
        "  acc",
        "",
        "train",
        "  loss",
    ]
    logger.close()


def test_section_headers_are_omitted_without_nesting():
    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.callback(state, logdict({"a": jnp.ones(1), "b": jnp.ones(1)}))
    assert names(logger) == ["a", "b"]
    logger.close()


def test_deeper_nesting_groups_on_the_first_segment():
    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.callback(state, logdict({"train": {"opt": {"lr": jnp.ones(1)}}}))
    assert names(logger) == [
        "train",
        "  opt/lr",
    ]
    logger.close()


def bars(logger) -> dict[str, tuple[float, float]]:
    """Maps each rendered bar's key to its ``(completed, total)``."""
    renderable = logger.live.get_renderable().renderable
    if not isinstance(renderable, Group):
        return {}
    keys, columns, _ = (column._cells for column in renderable.renderables[-1].columns)
    return {k: (bar.completed, bar.total) for k, bar in zip(keys, columns, strict=True)}


def test_progress_key_is_barred_and_kept_out_of_the_table():
    logger = ConsoleLogger(progress={"step": 10_000})
    state = logger.init(jax.random.key(0))
    logger.callback(
        state, logdict({"step": jnp.array([4200.0]), "loss": jnp.array([1.0])})
    )
    assert bars(logger) == {"step": (4200.0, 10_000)}
    # A blank row separates the metrics from the bars below them.
    assert [name for name in names(logger) if name] == ["loss"]
    logger.close()


def test_progress_uses_the_maximum_regardless_of_order():
    logger = ConsoleLogger(progress={"step": 100})
    state = logger.init(jax.random.key(0))
    logger.callback(state, logdict({"step": jnp.array([10.0, 40.0, 25.0])}))
    assert bars(logger)["step"] == (40.0, 100)
    logger.close()


def test_progress_averages_lockstep_runs():
    logger = ConsoleLogger(progress={"step": 100})
    for seed in range(3):
        state = logger.init(jax.random.key(seed))
        logger.callback(state, logdict({"step": jnp.array([30.0])}))
    assert bars(logger)["step"] == (30.0, 100)
    logger.close()


def test_no_bar_until_the_key_is_logged():
    logger = ConsoleLogger(progress={"step": 100})
    state = logger.init(jax.random.key(0))
    logger.callback(state, logdict({"loss": jnp.array([1.0])}))
    assert bars(logger) == {}
    logger.close()


def test_progress_is_absent_without_configuration():
    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.callback(state, logdict({"step": jnp.array([5.0])}))
    assert bars(logger) == {}
    assert list(rendered(logger)) == ["step"]
    logger.close()


@pytest.mark.parametrize(
    "value, expected",
    [
        (1234567.0, "1,234,567"),  # a step counter, not 1.235e+06
        (12345.6, "12,346"),
        (9999.0, "9999"),
        (412.5, "412.5"),
        (1e-9, "1e-09"),  # small values keep scientific notation
        (0.0, "0"),
        (-55123.0, "-55,123"),
    ],
)
def test_large_values_avoid_scientific_notation(value, expected):
    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.callback(state, logdict({"v": jnp.array([value])}))
    assert rendered(logger)["v"] == expected
    logger.close()


def test_complex_values_are_summarised_by_magnitude():
    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.callback(state, logdict({"z": jnp.array([3 + 4j])}))
    assert rendered(logger)["z"] == "5"
    logger.close()


def test_rendering_is_mutually_exclusive():
    # tap fires its callbacks through an unordered jax.debug.callback, so two can
    # land at once and one would mutate logss while the other iterates it. Rather
    # than trying to provoke that race -- the window is far too small to hit
    # reliably -- assert the exclusion that prevents it, by widening the critical
    # section until an overlap would be unmissable.
    overlaps = []

    class SlowConsoleLogger(ConsoleLogger):
        inside = False

        def _render(self, *args, **kwargs):
            if type(self).inside:
                overlaps.append(True)
            type(self).inside = True
            time.sleep(0.01)
            try:
                return super()._render(*args, **kwargs)
            finally:
                type(self).inside = False

    logger = SlowConsoleLogger()
    state = logger.init(jax.random.key(0))

    def report():
        for _ in range(5):
            logger.callback(state, logdict({"loss": jnp.ones(1)}))

    threads = [threading.Thread(target=report) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert overlaps == []
    logger.close()


def test_live_display_is_shared_across_runs():
    logger = ConsoleLogger()
    logger.init(jax.random.key(0))
    live = logger.live
    logger.init(jax.random.key(1))
    assert logger.live is live
    logger.close()
    assert logger.live is None
