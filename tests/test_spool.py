import jax
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
