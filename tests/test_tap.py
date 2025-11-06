import contextlib
import io

import jax
import pytest
from functions import *

import lox

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
