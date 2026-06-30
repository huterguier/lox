import jax
import jax.numpy as jnp
import pytest

import lox


def test_same_keys_no_unify():
    def f(x):
        def true_fn(x):
            lox.log({"x": x + 1})
            return x + 1

        def false_fn(x):
            lox.log({"x": x - 1})
            return x - 1

        return jax.lax.cond(x[0] > 0, true_fn, false_fn, x)

    x = jnp.ones(4)
    y, logs = lox.spool(f)(x)
    assert set(logs.keys()) == {"x"}
    assert jnp.allclose(y, x + 1)


def test_divergent_keys_raises_by_default():
    def f(x):
        def true_fn(x):
            lox.log({"a": x})
            return x + 1

        def false_fn(x):
            lox.log({"b": x})
            return x - 1

        return jax.lax.cond(x[0] > 0, true_fn, false_fn, x)

    with pytest.raises(ValueError, match="different keys"):
        lox.spool(f)(jnp.ones(4))


def test_shape_mismatch_always_raises():
    def f(x):
        def true_fn(x):
            lox.log({"v": x[:2]})
            return x + 1

        def false_fn(x):
            lox.log({"v": x[:3]})
            return x - 1

        return jax.lax.cond(x[0] > 0, true_fn, false_fn, x)

    with pytest.raises(ValueError, match="mismatched shapes"):
        lox.spool(f, unify=True)(jnp.ones(4))


def test_divergent_true_branch():
    def f(x):
        def true_fn(x):
            lox.log({"a": x, "b": x * 2})
            return x + 1

        def false_fn(x):
            lox.log({"a": x})
            return x - 1

        return jax.lax.cond(x[0] > 0, true_fn, false_fn, x)

    x = jnp.ones(4)
    y, logs = lox.spool(f, unify=True)(x)
    assert set(logs.keys()) == {"a", "b"}
    assert jnp.allclose(y, x + 1)
    assert jnp.allclose(logs["a"], x[None])
    assert jnp.allclose(logs["b"], (x * 2)[None])


def test_divergent_false_branch():
    def f(x):
        def true_fn(x):
            lox.log({"a": x, "b": x * 2})
            return x + 1

        def false_fn(x):
            lox.log({"a": x})
            return x - 1

        return jax.lax.cond(x[0] > 0, true_fn, false_fn, x)

    x = -jnp.ones(4)
    y, logs = lox.spool(f, unify=True)(x)
    assert set(logs.keys()) == {"a", "b"}
    assert jnp.allclose(y, x - 1)
    assert jnp.allclose(logs["a"], x[None])
    assert jnp.all(jnp.isnan(logs["b"]))


def test_one_branch_no_logs_true():
    def f(x):
        def true_fn(x):
            lox.log({"x": x})
            return x + 1

        def false_fn(x):
            return x - 1

        return jax.lax.cond(x[0] > 0, true_fn, false_fn, x)

    x = jnp.ones(4)
    y, logs = lox.spool(f, unify=True)(x)
    assert set(logs.keys()) == {"x"}
    assert jnp.allclose(logs["x"], x[None])


def test_one_branch_no_logs_false():
    def f(x):
        def true_fn(x):
            lox.log({"x": x})
            return x + 1

        def false_fn(x):
            return x - 1

        return jax.lax.cond(x[0] > 0, true_fn, false_fn, x)

    x = -jnp.ones(4)
    y, logs = lox.spool(f, unify=True)(x)
    assert set(logs.keys()) == {"x"}
    assert jnp.all(jnp.isnan(logs["x"]))


def test_int_fill():
    def f(x):
        def true_fn(x):
            lox.log({"i": jnp.array([1])})
            return x + 1

        def false_fn(x):
            return x - 1

        return jax.lax.cond(x[0] > 0, true_fn, false_fn, x)

    x = -jnp.ones(4)
    _, logs = lox.spool(f, unify=True)(x)
    assert logs["i"] == 0


def test_bool_fill():
    def f(x):
        def true_fn(x):
            lox.log({"flag": jnp.array([True])})
            return x + 1

        def false_fn(x):
            return x - 1

        return jax.lax.cond(x[0] > 0, true_fn, false_fn, x)

    x = -jnp.ones(4)
    _, logs = lox.spool(f, unify=True)(x)
    assert not logs["flag"]
