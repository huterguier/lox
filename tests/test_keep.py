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
def test_keep_output_unchanged(f, x):
    y_kept = lox.keep(f)(x)
    y_ref = f(x)
    assert jax.tree.all(jax.tree.map(jnp.allclose, y_kept, y_ref))


@pytest.mark.parametrize("f", functions)
def test_keep_noop_by_default(f, x):
    _, logs_kept = lox.spool(lox.keep(f))(x)
    _, logs_ref = lox.spool(f)(x)
    assert logs_kept.keys() == logs_ref.keys()
    assert jax.tree.all(jax.tree.map(jnp.allclose, logs_kept, logs_ref))


def _f_ab_train_c_eval(x):
    lox.log({"a": x, "b": x}, tags=("train",))
    lox.log({"c": x}, tags=("eval",))
    return x + 1


def test_keep_argnames():
    x = jnp.ones(4)
    _, logs = lox.spool(lox.keep(_f_ab_train_c_eval, argnames=["a"]))(x)
    assert set(logs.keys()) == {"a"}


def test_keep_tags():
    x = jnp.ones(4)
    _, logs = lox.spool(lox.keep(_f_ab_train_c_eval, tags=["train"]))(x)
    assert set(logs.keys()) == {"a", "b"}


def test_keep_argnames_and_tags_is_and():
    x = jnp.ones(4)
    _, logs = lox.spool(
        lox.keep(_f_ab_train_c_eval, argnames=["a"], tags=["train"])
    )(x)
    assert set(logs.keys()) == {"a"}


def test_keep_empty_argnames_keeps_nothing():
    x = jnp.ones(4)
    _, logs = lox.spool(lox.keep(_f_ab_train_c_eval, argnames=[]))(x)
    assert len(logs) == 0


def test_keep_empty_tags_keeps_nothing():
    x = jnp.ones(4)
    _, logs = lox.spool(lox.keep(_f_ab_train_c_eval, tags=[]))(x)
    assert len(logs) == 0
