import jax
import jax.numpy as jnp
import pytest
from functions import *

import lox

functions = [
    (f_id, f_id_spooled),
    (f_add, f_add_spooled),
    (f_scan, f_scan_spooled),
    (f_call, f_call_spooled),
    (f_jit, f_jit_spooled),
    (f_cond, f_cond_spooled),
    (f_grad, f_grad_spooled),
]


@pytest.fixture(params=[0, 1, 2])
def key(request):
    return jax.random.key(request.param)


@pytest.fixture(params=[(4,), (2, 3), (5, 5)])
def x(request, key):
    shape = request.param
    return jax.random.normal(key, shape)


@pytest.mark.parametrize("f, f_spooled", functions)
def test_spool(f, f_spooled, x):
    y_f, logs_f = lox.spool(f)(x)
    y_f_spooled, logs_f_spooled = f_spooled(x)
    assert jax.tree.all(
        jax.tree.map(lambda a, b: jax.numpy.allclose(a, b), y_f, y_f_spooled)
    )
    assert jax.tree.all(
        jax.tree.map(lambda a, b: jax.numpy.allclose(a, b), logs_f, logs_f_spooled)
    )


@pytest.mark.parametrize("f, f_spooled", functions)
def test_spool_argnames(f, f_spooled, x):
    argnames = ["x"]
    y_f, logs_f = lox.spool(f, argnames=argnames)(x)
    y_f_spooled, logs_f_spooled = f_spooled(x)
    logs_f_spooled = logs_f_spooled.filter(lambda k, _: k in argnames)
    assert jax.tree.all(
        jax.tree.map(lambda a, b: jax.numpy.allclose(a, b), y_f, y_f_spooled)
    )
    assert jax.tree.all(
        jax.tree.map(lambda a, b: jax.numpy.allclose(a, b), logs_f, logs_f_spooled)
    )


def test_spool_cond_divergent_branches_requires_default():
    def f(x):
        return jax.lax.cond(
            x > 0,
            lambda v: (lox.log({"val": v}), v)[1],
            lambda v: v,
            x,
        )

    with pytest.raises(
        ValueError, match="Divergent logging branches detected in jax.lax.cond"
    ):
        lox.spool(f)(1.0)


def test_spool_cond_divergent_branches_with_default():
    def f(x):
        return jax.lax.cond(
            x > 0,
            lambda v: (lox.log({"val": v}, default={"val": jnp.nan}), v)[1],
            lambda v: v,
            x,
        )

    _, logs_true = lox.spool(f)(1.0)
    _, logs_false = lox.spool(f)(-1.0)
    assert jnp.allclose(logs_true["val"], jnp.array([1.0]))
    assert jnp.isnan(logs_false["val"]).all()


def test_spool_cond_extra_log_requires_default_for_extra_position():
    def f(x):
        def true_fun(v):
            lox.log({"val": v})
            return v

        def false_fun(v):
            lox.log({"val": v})
            lox.log({"val": v * 2}, default={"val": -1.0})
            return v

        return jax.lax.cond(x > 0, true_fun, false_fun, x)

    _, logs = lox.spool(f)(1.0)
    assert jnp.allclose(logs["val"], jnp.array([1.0, -1.0]))
