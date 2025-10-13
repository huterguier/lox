import os
import shutil

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

import lox
from lox.loggers import MultiLogger, SaveLogger
from lox.wandb import WandbLogger


def f(xs):
    def step(carry, x):
        lox.log({"x": x, "carry": carry})
        return carry + x, carry

    return jax.lax.scan(step, 0, xs)


def f_step(xs):
    def step(carry, x):
        lox.log({"x": x, "carry": carry}, step=x)
        return carry + x, carry

    return jax.lax.scan(step, 0, xs)


class SpoolTest(parameterized.TestCase):
    def test_save(self):
        key = jax.random.PRNGKey(0)
        logger = SaveLogger("./.lox")
        logger_state = logger.init(key)

        xs = jnp.arange(10)
        _ = logger.spool(f, logger_state)(xs)

    def test_wandb(self):
        key = jax.random.PRNGKey(0)
        logger = WandbLogger(project="lox", name="test_wandb")
        logger_state = logger.init(key)

        xs = jnp.arange(100)
        _ = logger.spool(f_step, logger_state, reduce="last")(xs)

    def test_multi(self):
        key = jax.random.PRNGKey(1)
        logger = MultiLogger(SaveLogger("./.lox"), WandbLogger(project="lox"))
        logger_state = logger.init(key)

        xs = jnp.arange(100)
        _ = logger.spool(f_step, logger_state, interval=10)(xs)


if __name__ == "__main__":
    if os.path.exists("./.lox"):
        shutil.rmtree("./.lox")
    absltest.main()
