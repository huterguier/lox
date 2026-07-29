import jax
import jax.numpy as jnp
import pytest
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


def rendered(logger) -> dict[str, str]:
    """Maps each rendered row's key to its ``mean ± std`` cell."""
    table = logger.live.get_renderable()
    keys, values, _ = (column._cells for column in table.columns)
    return dict(zip(keys, values, strict=True))


def details(logger) -> dict[str, str]:
    """Maps each rendered row's key to its trailing detail cell."""
    table = logger.live.get_renderable()
    keys, _, cells = (column._cells for column in table.columns)
    return dict(zip(keys, cells, strict=True))


def test_divergent_keys_are_all_rendered():
    logger = ConsoleLogger()
    for seed, logs in enumerate(
        [{"loss": jnp.array([5.0])}, {"accuracy": jnp.array([5.0])}]
    ):
        state = logger.init(jax.random.key(seed))
        logger.callback(state, logdict(logs))
    assert set(rendered(logger)) == {"[bold]loss[/bold]", "[bold]accuracy[/bold]"}
    logger.close()


def test_key_is_pooled_over_runs_that_logged_it():
    logger = ConsoleLogger()
    for seed, value in enumerate([5.0, 7.0]):
        state = logger.init(jax.random.key(seed))
        logger.callback(state, logdict({"loss": jnp.array([value])}))
    assert rendered(logger)["[bold]loss[/bold]"] == "6 ± 1"
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
    assert set(rendered(logger)) == {"[bold]m/a[/bold]"}
    logger.close()


def test_callback_without_init_does_not_raise():
    other = ConsoleLogger()
    state = other.init(jax.random.key(0))
    other.close()

    logger = ConsoleLogger()
    logger.callback(state, logdict({"x": jnp.ones(2)}))
    assert set(rendered(logger)) == {"[bold]x[/bold]"}
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
    assert rendered(logger)["[bold]loss[/bold]"] == "4.85 ± 0.04082"
    logger.close()


def test_single_run_deviation_covers_all_values():
    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.callback(state, logdict({"loss": jnp.array([10.0, 7.0, 4.0, 2.0, 1.0])}))
    assert rendered(logger)["[bold]loss[/bold]"] == "4.8 ± 3.311"
    logger.close()


def test_deviation_is_omitted_for_a_lone_value():
    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.callback(state, logdict({"loss": jnp.array([5.0])}))
    assert rendered(logger)["[bold]loss[/bold]"] == "5"

    # A second value to compare against brings the deviation back, even when the
    # values agree and it is therefore zero.
    state = logger.init(jax.random.key(1))
    logger.callback(state, logdict({"loss": jnp.array([5.0])}))
    assert rendered(logger)["[bold]loss[/bold]"] == "5 ± 0"
    logger.close()


def test_deviation_is_kept_for_a_single_run_with_several_values():
    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.callback(state, logdict({"loss": jnp.array([4.0, 6.0])}))
    assert rendered(logger)["[bold]loss[/bold]"] == "5 ± 1"
    logger.close()


def test_detail_reports_run_count_and_shape():
    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.callback(state, logdict({"img": jnp.ones((5, 3, 4))}))
    assert details(logger)["[bold]img[/bold]"] == "[dim]1 run, (5, 3, 4)[/dim]"

    state = logger.init(jax.random.key(1))
    logger.callback(state, logdict({"img": jnp.ones((5, 3, 4))}))
    assert details(logger)["[bold]img[/bold]"] == "[dim]2 runs, each (5, 3, 4)[/dim]"
    logger.close()


def test_detail_reports_partial_and_mixed_shapes():
    logger = ConsoleLogger()
    for seed, shape in enumerate([(5,), (3,), (5,)]):
        state = logger.init(jax.random.key(seed))
        logs = {"loss": jnp.ones(shape)}
        if seed == 0:
            logs["only_first"] = jnp.ones(2)
        logger.callback(state, logdict(logs))
    assert details(logger)["[bold]loss[/bold]"] == "[dim]3 runs, mixed shapes[/dim]"
    assert details(logger)["[bold]only_first[/bold]"] == "[dim]1 run, (2,)[/dim]"
    logger.close()


def test_vmapped_init_yields_one_run_per_lane():
    def f(x):
        lox.log({"v": x.sum()})
        return x

    logger = ConsoleLogger()
    keys = jnp.stack([jax.random.key(seed) for seed in range(3)])
    states = jax.vmap(logger.init)(keys)
    jax.vmap(lambda state, x: logger.spool(f, state)(x))(states, jnp.ones((3, 5)))
    assert details(logger)["[bold]v[/bold]"] == "[dim]3 runs, each (1,)[/dim]"
    logger.close()


def test_shared_state_across_vmap_lanes_is_a_single_run():
    def f(x):
        lox.log({"v": x.sum()})
        return x

    logger = ConsoleLogger()
    state = logger.init(jax.random.key(0))
    logger.spool(jax.vmap(f), state)(jnp.ones((3, 5)))
    assert details(logger)["[bold]v[/bold]"] == "[dim]1 run, (3, 1)[/dim]"
    logger.close()


def test_live_display_is_shared_across_runs():
    logger = ConsoleLogger()
    logger.init(jax.random.key(0))
    live = logger.live
    logger.init(jax.random.key(1))
    assert logger.live is live
    logger.close()
    assert logger.live is None
