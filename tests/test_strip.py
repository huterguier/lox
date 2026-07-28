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


def _f_ab_train_c_eval(x):
    lox.log({"a": x, "b": x}, tags=("train",))
    lox.log({"c": x}, tags=("eval",))
    return x + 1


def test_strip_argnames():
    x = jnp.ones(4)
    _, logs = lox.spool(lox.strip(_f_ab_train_c_eval, argnames=["a"]))(x)
    assert set(logs.keys()) == {"b", "c"}


def test_strip_tags():
    x = jnp.ones(4)
    _, logs = lox.spool(lox.strip(_f_ab_train_c_eval, tags=["train"]))(x)
    assert set(logs.keys()) == {"c"}


def test_strip_argnames_and_tags_is_and():
    x = jnp.ones(4)
    _, logs = lox.spool(lox.strip(_f_ab_train_c_eval, argnames=["a"], tags=["train"]))(
        x
    )
    # only "a" matches both argnames and tags; "b" is tagged but not named,
    # "c" is named-excluded and not tagged -- neither should be stripped.
    assert set(logs.keys()) == {"b", "c"}


def test_strip_empty_argnames_strips_nothing():
    x = jnp.ones(4)
    _, logs = lox.spool(lox.strip(_f_ab_train_c_eval, argnames=[]))(x)
    assert set(logs.keys()) == {"a", "b", "c"}


def test_strip_empty_tags_strips_nothing():
    x = jnp.ones(4)
    _, logs = lox.spool(lox.strip(_f_ab_train_c_eval, tags=[]))(x)
    assert set(logs.keys()) == {"a", "b", "c"}


def test_strip_bare_string_argnames_is_exact_match():
    x = jnp.ones(4)
    _, logs = lox.spool(lox.strip(_f_ab_train_c_eval, argnames="a"))(x)
    assert set(logs.keys()) == {"b", "c"}


def test_strip_bare_string_tags_is_exact_match():
    x = jnp.ones(4)
    _, logs = lox.spool(lox.strip(_f_ab_train_c_eval, tags="train"))(x)
    assert set(logs.keys()) == {"c"}
