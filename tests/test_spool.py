import contextlib
import io

import jax
import jax.numpy as jnp
import pytest
from functions import f_add, f_call, f_cond, f_grad, f_id, f_jit, f_remat, f_scan, f_while

import lox

functions = [
    (f_id,    {"x"}),
    (f_add,   {"x", "z"}),
    (f_scan,  {"carry", "x"}),
    (f_call,  {"x"}),
    (f_jit,   {"x"}),
    (f_cond,  {"branch", "x"}),
    (f_grad,  {"x"}),
    (f_remat, {"x"}),
]


@pytest.fixture(params=[0, 1, 2])
def key(request):
    return jax.random.key(request.param)


@pytest.fixture(params=[(4,), (2, 3), (5, 5)])
def x(request, key):
    return jax.random.normal(key, request.param)


@pytest.mark.parametrize("f, expected_keys", functions)
def test_spool_output_unchanged(f, expected_keys, x):
    y_spooled, _ = lox.spool(f)(x)
    y_ref = f(x)
    assert jax.tree.all(jax.tree.map(jnp.allclose, y_spooled, y_ref))


@pytest.mark.parametrize("f, expected_keys", functions)
def test_spool_keys(f, expected_keys, x):
    _, logs = lox.spool(f)(x)
    assert set(logs.keys()) == expected_keys


@pytest.mark.parametrize("f, expected_keys", functions)
def test_spool_argnames(f, expected_keys, x):
    _, logs = lox.spool(f, argnames=["x"])(x)
    assert set(logs.keys()) == {"x"}


def test_spool_while_warns(x):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        _, logs = lox.spool(f_while)(x)
    assert "Warning" in out.getvalue()
    assert len(logs) == 0


def test_spool_vmap():
    x_batch = jnp.ones((3, 4))
    _, logs = jax.vmap(lox.spool(f_id))(x_batch)
    assert logs["x"].shape == (3, 1, 4)


def test_spool_interval():
    x = jnp.ones(100)
    _, logs = lox.spool(f_scan, interval=10)(x)
    assert logs["x"].shape[0] == 10


def test_spool_reduce():
    x = jnp.ones(10)
    _, logs = lox.spool(f_scan, reduce="mean")(x)
    assert logs["x"].shape[0] == 1


def test_spool_prefix():
    x = jnp.ones(4)
    _, logs = lox.spool(f_id, prefix="train/")(x)
    assert "train/x" in logs
    assert "x" not in logs


def test_spool_tags():
    def f(x):
        lox.log({"a": x}, tags=("train",))
        lox.log({"b": x}, tags=("eval",))
        return x

    x = jnp.ones(4)
    _, logs = lox.spool(f, tags=["train"])(x)
    assert "a" in logs
    assert "b" not in logs
