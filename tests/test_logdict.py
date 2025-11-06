import jax
import jax.numpy as jnp
import pytest

import lox


def test_or():
    def f(xs):
        def step(carry, x):
            lox.log({"x": x, "carry": carry})
            return carry + x, carry

        return jax.lax.scan(step, 0, xs)

    xs = jnp.arange(100)
    _, logs = lox.spool(f)(xs)


def test_slice():
    def f(xs):
        def step(carry, x):
            lox.log({"x": x, "carry": carry}, step=x)
            return carry + x, carry

        return jax.lax.scan(step, 0, xs)

    xs = jnp.arange(100)
    _, logs = lox.spool(f)(xs)
    logs_sliced = logs.slice[::10]
    assert logs_sliced["x"].shape[0] == 10


def test_getattr():
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


def test_add():
    def g(x):
        lox.log({"x": x})
        return x + 1

    def f(x):
        return g(x)

    x = jnp.array(3)
    _, logs = lox.spool(f)(x)
    assert logs["x"] == x
