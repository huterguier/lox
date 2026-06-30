import contextlib
import io

import jax
import jax.numpy as jnp
import pytest
from functions import f_add, f_call, f_cond, f_grad, f_id, f_jit, f_remat, f_scan

import lox
from lox import logdict

functions = [f_id, f_add, f_scan, f_call, f_jit, f_cond, f_grad, f_remat]


@pytest.fixture(params=[0, 1, 2])
def key(request):
    return jax.random.key(request.param)


@pytest.fixture(params=[(4,), (2, 3), (5, 5)])
def x(request, key):
    shape = request.param
    return jax.random.normal(key, shape)


@pytest.mark.parametrize("f", functions)
def test_tap(f, x):
    with contextlib.redirect_stdout(io.StringIO()) as f_stdout:
        _ = lox.tap(f)(x)
        output = f_stdout.getvalue()
    assert output.strip() != ""


@pytest.mark.parametrize("f", functions)
def test_tap_spool_equivalence(f, x):
    global logs_tap
    logs_tap = logdict({})

    def callback(logs):
        global logs_tap
        logs_tap = logs_tap + logs

    _ = lox.tap(f, callback=callback)(x)
    _, logs_spool = lox.spool(f)(x)

    assert logs_tap.keys() == logs_spool.keys()
    assert jax.tree.all(jax.tree.map(jnp.allclose, logs_tap, logs_spool))
