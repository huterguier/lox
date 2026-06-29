"""Tests for cond branch log unification (issue #7)."""
import jax
import jax.numpy as jnp
import pytest

import lox


def _cond_divergent(x):
    """Branches log different keys."""
    def true_branch(x):
        lox.log({"x": x, "y": x + 1.0})
        return x + 1.0

    def false_branch(x):
        lox.log({"x": x, "z": x - 1.0})
        return x - 1.0

    return jax.lax.cond(x > 0, true_branch, false_branch, x)


def _cond_one_logs(x):
    """Only one branch logs."""
    def true_branch(x):
        lox.log({"x": x})
        return x + 1.0

    def false_branch(x):
        return x - 1.0

    return jax.lax.cond(x > 0, true_branch, false_branch, x)


def _cond_shape_mismatch(x):
    """Both branches log the same key but with different shapes."""
    def true_branch(x):
        lox.log({"val": x})
        return x + 1.0

    def false_branch(x):
        lox.log({"val": jnp.stack([x, x])})
        return x - 1.0

    return jax.lax.cond(x > 0, true_branch, false_branch, x)


# --- error cases ---

def test_divergent_keys_raises_by_default():
    with pytest.raises(ValueError, match="different keys"):
        lox.spool(_cond_divergent)(jnp.array(1.0))


def test_one_branch_no_logs_raises_by_default():
    with pytest.raises(ValueError, match="different keys"):
        lox.spool(_cond_one_logs)(jnp.array(1.0))


def test_shape_mismatch_always_raises():
    with pytest.raises(ValueError, match="mismatched shape"):
        lox.spool(_cond_shape_mismatch, unify=True)(jnp.array(1.0))


def test_shape_mismatch_raises_without_unify():
    with pytest.raises(ValueError, match="mismatched shape"):
        lox.spool(_cond_shape_mismatch)(jnp.array(1.0))


# --- unify=True: correct values from taken branch ---

def test_divergent_unify_true_branch():
    _, logs = lox.spool(_cond_divergent, unify=True)(jnp.array(2.0))
    assert jnp.allclose(logs["x"], jnp.array(2.0))
    assert jnp.allclose(logs["y"], jnp.array(3.0))
    assert jnp.isnan(logs["z"])


def test_divergent_unify_false_branch():
    _, logs = lox.spool(_cond_divergent, unify=True)(jnp.array(-1.0))
    assert jnp.allclose(logs["x"], jnp.array(-1.0))
    assert jnp.isnan(logs["y"])
    assert jnp.allclose(logs["z"], jnp.array(-2.0))


def test_one_branch_no_logs_unify_true_branch():
    _, logs = lox.spool(_cond_one_logs, unify=True)(jnp.array(1.0))
    assert jnp.allclose(logs["x"], jnp.array(1.0))


def test_one_branch_no_logs_unify_false_branch():
    _, logs = lox.spool(_cond_one_logs, unify=True)(jnp.array(-1.0))
    assert jnp.isnan(logs["x"])


# --- unify=True: integer fill is 0 ---

def test_divergent_int_fill():
    def f(x):
        def true_branch(x):
            lox.log({"n": jnp.int32(1), "x": x})
            return x

        def false_branch(x):
            lox.log({"x": x})
            return x

        return jax.lax.cond(x > 0, true_branch, false_branch, x)

    _, logs = lox.spool(f, unify=True)(jnp.array(-1.0))
    assert logs["n"] == 0


# --- unify=True: bool fill is False ---

def test_divergent_bool_fill():
    def f(x):
        def true_branch(x):
            lox.log({"flag": jnp.bool_(True), "x": x})
            return x

        def false_branch(x):
            lox.log({"x": x})
            return x

        return jax.lax.cond(x > 0, true_branch, false_branch, x)

    _, logs = lox.spool(f, unify=True)(jnp.array(-1.0))
    assert logs["flag"] == False


# --- non-divergent cond still works ---

def test_same_keys_no_unify():
    def f(x):
        def true_branch(x):
            lox.log({"x": x})
            return x + 1.0

        def false_branch(x):
            lox.log({"x": x * 2.0})
            return x - 1.0

        return jax.lax.cond(x > 0, true_branch, false_branch, x)

    _, logs_t = lox.spool(f)(jnp.array(3.0))
    _, logs_f = lox.spool(f)(jnp.array(-2.0))
    assert jnp.allclose(logs_t["x"], jnp.array(3.0))
    assert jnp.allclose(logs_f["x"], jnp.array(-4.0))


# --- nested cond inherits unify ---

def test_nested_cond_unify():
    def f(x):
        def outer_true(x):
            def inner_true(x):
                lox.log({"a": x, "b": x + 1.0})
                return x

            def inner_false(x):
                lox.log({"a": x})
                return x

            return jax.lax.cond(x > 5, inner_true, inner_false, x)

        def outer_false(x):
            lox.log({"a": x * 0.0})
            return x

        return jax.lax.cond(x > 0, outer_true, outer_false, x)

    _, logs = lox.spool(f, unify=True)(jnp.array(3.0))
    assert "a" in logs
    assert "b" in logs
