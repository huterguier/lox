import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

import lox


class LogdictTest(parameterized.TestCase):
    def test_or(self):
        def f(xs):
            def step(carry, x):
                lox.log({"x": x, "carry": carry})
                return carry + x, carry

            return jax.lax.scan(step, 0, xs)

        xs = jnp.arange(100)
        _, logs = lox.spool(f)(xs)


    def test_slice(self):
        def f(xs):
            def step(carry, x):
                lox.log({"x": x, "carry": carry}, step=x)
                return carry + x, carry

            return jax.lax.scan(step, 0, xs)

        xs = jnp.arange(100)
        _, logs = lox.spool(f)(xs)
        print(logs.step)
        logs_sliced = logs.slice[::10]
        print(logs_sliced)
        print(logs_sliced.step)
        assert logs_sliced["x"].shape[0] == 10


    def test_getattr(self):
        def f(x):
            def true_fun(x):
                lox.log({"x": x, "branch": True})
                return x + 1
            def false_fun(x):
                lox.log({"x": x, "branch": False})
                return x - 1
            return jax.lax.cond(x > 5, true_fun, false_fun, x)
        x = jnp.array(3)
        _, logs = lox.spool(f)(x)

        assert logs["x"] == x
        assert not logs["branch"]


    def test_add(self):
        def g(x):
            lox.log({"x": x})
            return x + 1
        def f(x):
            return g(x)

        x = jnp.array(3)
        _, logs = lox.spool(f)(x)
        assert logs["x"] == x


if __name__ == "__main__":
    absltest.main()

