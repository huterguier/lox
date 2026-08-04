from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest
from test_logger import TestLogger

import lox
from lox.wandb import WandbLogger


@pytest.fixture(autouse=True)
def mock_wandb(monkeypatch):
    """Stand in for the real wandb backend so tests never hit the network."""
    runs = []

    def fake_init(*args, **kwargs):
        run = MagicMock()
        run.id = f"run-{len(runs)}"
        runs.append(run)
        return run

    monkeypatch.setattr("wandb.init", fake_init)
    return runs


class TestWandbLogger(TestLogger):
    @pytest.fixture
    def logger(self):
        return WandbLogger(project="test")


def test_init_creates_a_wandb_run(mock_wandb):
    logger = WandbLogger(project="test")
    logger.init(jax.random.key(0))
    assert len(mock_wandb) == 1


def test_spool_logs_data_to_the_run(mock_wandb):
    logger = WandbLogger(project="test")
    state = logger.init(jax.random.key(0))

    def f(x):
        lox.log({"loss": x})
        return x + 1

    logger.spool(f, state)(jnp.array(1.0))

    run = mock_wandb[0]
    run.log.assert_called_once_with({"loss": jnp.array(1.0)})
