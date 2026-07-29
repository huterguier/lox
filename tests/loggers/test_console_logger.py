import jax
import jax.numpy as jnp
import pytest
from test_logger import TestLogger

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


def test_live_display_is_shared_across_runs():
    logger = ConsoleLogger()
    logger.init(jax.random.key(0))
    live = logger.live
    logger.init(jax.random.key(1))
    assert logger.live is live
    logger.close()
    assert logger.live is None
