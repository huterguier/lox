import jax
import jax.numpy as jnp
import shutil
import os
from absl.testing import absltest, parameterized

import lox
from lox.save import SaveLogger
from lox.logger import MultiLogger
from lox.wandb.logger import WandbLogger


class SpoolTest(parameterized.TestCase):
    def test_save(self):
        def f(xs):
            def step(carry, x):
                lox.log({"x": x, "carry": carry})
                return carry + x, carry

            return jax.lax.scan(step, 0, xs)

        key = jax.random.PRNGKey(0)
        logger = SaveLogger("./.lox")
        logger_state = logger.init(key)

        xs = jnp.arange(10)
        ys = logger.spool(logger_state, f)(xs)

    def test_wandb(self):
        def f(xs):
            def step(carry, x):
                lox.log({"x": x, "carry": carry})
                return carry + x, carry

            return jax.lax.scan(step, 0, xs)

        key = jax.random.PRNGKey(0)
        logger = WandbLogger(project="lox", name="test_wandb")
        logger_state = logger.init(key)

        xs = jnp.arange(10)
        ys = logger.spool(logger_state, f, interval=2)(xs)

        ys, logs = lox.spool(f)(xs)
        logs_sliced = logs.slice[::2]
        print(logs_sliced)

    def test_multi(self):
        def f(xs):
            def step(carry, x):
                lox.log({"x": x, "carry": carry})
                return carry + x, carry

            return jax.lax.scan(step, 0, xs)

        key = jax.random.PRNGKey(1)
        logger = MultiLogger([SaveLogger("./.lox"), WandbLogger(project="lox")])
        logger_state = logger.init(key)

        xs = jnp.arange(10)
        ys = logger.spool(logger_state, f)(xs)


if __name__ == "__main__":
    if os.path.exists("./.lox"):
        shutil.rmtree("./.lox")
    absltest.main()
