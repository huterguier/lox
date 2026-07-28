import jax
import jax.numpy as jnp
import pytest

import lox


@pytest.fixture
def scan_logs():
    def f(xs):
        def step(carry, x):
            lox.log({"x": x, "carry": carry})
            return carry + x, carry

        return jax.lax.scan(step, 0.0, xs)

    xs = jnp.arange(10, dtype=float)
    _, logs = lox.spool(f)(xs)
    return logs


def test_slice(scan_logs):
    sliced = scan_logs.slice[::2]
    assert sliced["x"].shape[0] == 5


def test_add():
    x = jnp.ones(4)
    _, logs1 = lox.spool(lambda x: (lox.log({"x": x}), x)[-1])(x)
    _, logs2 = lox.spool(lambda x: (lox.log({"x": x * 2}), x)[-1])(x)
    combined = logs1 + logs2
    assert combined["x"].shape[0] == 2
    assert jnp.allclose(combined["x"][0], x)
    assert jnp.allclose(combined["x"][1], x * 2)


def test_or():
    x = jnp.ones(4)

    def f1(x):
        lox.log({"a": x})
        return x

    def f2(x):
        lox.log({"a": x * 2, "b": x})
        return x

    _, logs1 = lox.spool(f1)(x)
    _, logs2 = lox.spool(f2)(x)
    merged = logs1 | logs2
    assert set(merged.keys()) == {"a", "b"}
    assert jnp.allclose(merged["a"], logs2["a"])


def test_reduce_mean(scan_logs):
    reduced = scan_logs.reduce("mean")
    for v in reduced.values():
        assert v.shape[0] == 1
    assert jnp.allclose(reduced["x"], scan_logs["x"].mean(keepdims=True))


def test_reduce_first(scan_logs):
    reduced = scan_logs.reduce("first")
    for v in reduced.values():
        assert v.shape[0] == 1
    assert jnp.allclose(reduced["x"], scan_logs["x"][:1])


def test_reduce_last(scan_logs):
    reduced = scan_logs.reduce("last")
    for v in reduced.values():
        assert v.shape[0] == 1
    assert jnp.allclose(reduced["x"], scan_logs["x"][-1:])


def test_filter(scan_logs):
    filtered = scan_logs.filter(lambda k, _: k == "x")
    assert set(filtered.keys()) == {"x"}
    assert "carry" not in filtered
    assert jnp.allclose(filtered["x"], scan_logs["x"])


def test_prefix(scan_logs):
    prefixed = scan_logs.prefix("train/")
    assert set(prefixed.keys()) == {"train/x", "train/carry"}
    assert jnp.allclose(prefixed["train/x"], scan_logs["x"])


def test_getattr():
    def f(x):
        def true_fun(x):
            lox.log({"x": x, "branch": True})
            return x + 1

        def false_fun(x):
            lox.log({"x": x, "branch": False})
            return x - 1

        return jax.lax.cond(x > 5, true_fun, false_fun, x)

    x = jnp.array(3.0)
    _, logs = lox.spool(f)(x)
    assert jnp.allclose(logs["x"], x)
    assert not logs["branch"]
