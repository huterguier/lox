import jax
import jax.numpy as jnp
import pytest
from functions import f_add, f_call, f_cond, f_grad, f_id, f_jit, f_remat, f_scan

import lox

functions = [f_id, f_add, f_scan, f_call, f_jit, f_cond, f_grad, f_remat]


@pytest.fixture(params=[0, 1, 2])
def key(request):
    return jax.random.key(request.param)


@pytest.fixture(params=[(4,), (2, 3), (5, 5)])
def x(request, key):
    return jax.random.normal(key, request.param)


@pytest.mark.parametrize("f", functions)
def test_strip_output_unchanged(f, x):
    y_stripped = lox.strip(f)(x)
    y_ref = f(x)
    assert jax.tree.all(jax.tree.map(jnp.allclose, y_stripped, y_ref))


@pytest.mark.parametrize("f", functions)
def test_strip_removes_logs(f, x):
    _, logs = lox.spool(lox.strip(f))(x)
    assert len(logs) == 0
