import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

import lox


class LogdictTest(parameterized.TestCase):
    def test_or(self):
        data = {
            "x": jax.random.normal(
        }

        logs1 = lox.log(

        assert jnp.equal(logs["x"], xs).all()
        assert jnp.equal(logs["carry"], ys[1]).all()


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

