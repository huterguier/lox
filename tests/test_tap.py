import jax
import jax.numpy as jnp
import io
import contextlib
from absl.testing import absltest, parameterized

import lox


class TapTest(parameterized.TestCase):
    def test_scan(self):
        @jax.jit
        def f(xs):
            def step(carry, x):
                lox.log({"x": x, "carry": carry})
                return carry + x, carry
            return jax.lax.scan(step, 0, xs)
        xs = jnp.arange(10)
        # buf = io.StringIO()
        # with contextlib.redirect_stdout(buf):
        ys, _ = lox.tap(f, argnames=("x",))(xs)


    def test_cond(self):
        @jax.jit
        def f(x):
            def true_fun(x):
                lox.log({"x": x, "branch": True})
                return x + 1
            def false_fun(x):
                lox.log({"x": x, "branch": False})
                return x - 1
            return jax.lax.cond(x > 5, true_fun, false_fun, x)
        x = jnp.array(3)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _ = lox.tap(f)(x)


    def test_call(self):
        def g(x):
            lox.log({"x": x})
            return x + 1
        @jax.jit
        def f(x):
            return g(x)

        x = jnp.array(3)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _ = lox.tap(f)(x)


if __name__ == "__main__":
    absltest.main()
