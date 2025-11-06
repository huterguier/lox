import contextlib
import io

import jax
import jax.numpy as jnp
import pytest

import lox


def f_id(x):
    lox.log({"x": x})
    return x


def f_add(x):
    z = x + x
    lox.log({"x": x, "z": z})
    return z


def f_scan(x):
    def step(carry, x):
        carry = carry + x.mean()
        lox.log({"carry": carry, "x": x})
        return carry, carry

    return jax.lax.scan(step, 0, x)


def f_call(x):
    def g(x):
        lox.log({"x": x})
        return x * 2

    return g(x) + 1


def f_jit(x):
    @jax.jit
    def g(x):
        lox.log({"x": x})
        return x * 3

    return g(x) + 1


def f_cond(x):
    def true_fun(x):
        x = x + 1
        lox.log({"branch": True, "x": x})
        return x

    def false_fun(x):
        x = x - 1
        lox.log({"branch": False, "x": x})
        return x

    cond = x.ravel()[0] > 0
    return jax.lax.cond(cond, true_fun, false_fun, x)


def f_grad(x):
    def func(x):
        lox.log({"x": x})
        return x.mean()

    grad_func = jax.grad(func)
    return grad_func(x)


functions = [
    (f_id),
    (f_add),
    (f_scan),
    (f_call),
    (f_jit),
    (f_cond),
    (f_grad),
]


@pytest.fixture(params=[0, 1, 2])
def key(request):
    return jax.random.key(request.param)


@pytest.fixture(params=[(4,), (2, 3), (5, 5)])
def x(request, key):
    shape = request.param
    return jax.random.normal(key, shape)


@pytest.mark.parametrize("f", functions)
def test_spool(f, x):
    with contextlib.redirect_stdout(io.StringIO()) as f_stdout:
        _ = lox.tap(f)(x)
        output = f_stdout.getvalue()
        print(output)
    assert output.strip() != ""
