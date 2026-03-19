"""Regression tests for jit trace cache corruption.

When spool/tap/strip are applied to a jax.jit-wrapped function, the inner
jaxpr objects are shared with jax.jit's trace cache. Previously, spool_jaxpr,
tap_jaxpr, and strip_jaxpr mutated these shared objects in place, corrupting
the cache and causing errors on subsequent calls.
"""

import contextlib
import io

import jax
import jax.numpy as jnp

import lox


def test_spool_jit_repeated_calls():
    """spool(jax.jit(f)) should work when called multiple times."""

    def f(x):
        def step(carry, _):
            carry = carry + 1.0
            lox.log({"carry": carry})
            return carry, None

        carry, _ = jax.lax.scan(step, x, length=5)
        return carry

    spooled = lox.spool(jax.jit(f))

    result1, logs1 = spooled(1.0)
    result2, logs2 = spooled(result1)

    assert float(result1) == 6.0
    assert float(result2) == 11.0
    assert logs1["carry"].shape == (5,)
    assert logs2["carry"].shape == (5,)


def test_spool_jit_simple():
    """spool(jax.jit(f)) with a simple log should work when called twice."""

    def f(x):
        y = x + 1.0
        lox.log({"y": y})
        return y

    spooled = lox.spool(jax.jit(f))

    result1, logs1 = spooled(1.0)
    result2, logs2 = spooled(result1)

    assert float(result1) == 2.0
    assert float(result2) == 3.0


def test_tap_jit_repeated_calls():
    """tap(jax.jit(f)) should work when called multiple times."""

    def f(x):
        y = x + 1.0
        lox.log({"y": y})
        return y

    collected = []

    def callback(logs):
        collected.append(dict(logs))

    tapped = lox.tap(jax.jit(f), callback=callback)

    result1 = tapped(1.0)
    result2 = tapped(result1)

    assert float(result1) == 2.0
    assert float(result2) == 3.0
    assert len(collected) == 2


def test_strip_jit_repeated_calls():
    """strip(jax.jit(f)) should work when called multiple times."""

    def f(x):
        y = x + 1.0
        lox.log({"y": y})
        return y

    stripped = lox.strip(jax.jit(f))

    result1 = stripped(1.0)
    result2 = stripped(result1)

    assert float(result1) == 2.0
    assert float(result2) == 3.0


def test_spool_jit_scan_repeated_calls():
    """spool(jax.jit(f)) with scan should work across multiple calls."""

    def f(xs):
        def step(carry, x):
            carry = carry + x
            lox.log({"carry": carry})
            return carry, carry

        carry, ys = jax.lax.scan(step, 0.0, xs)
        return ys

    spooled = lox.spool(jax.jit(f))

    xs1 = jnp.array([1.0, 2.0, 3.0])
    xs2 = jnp.array([4.0, 5.0, 6.0])

    result1, logs1 = spooled(xs1)
    result2, logs2 = spooled(xs2)

    assert jnp.allclose(result1, jnp.array([1.0, 3.0, 6.0]))
    assert jnp.allclose(result2, jnp.array([4.0, 9.0, 15.0]))
    assert logs1["carry"].shape == (3,)
    assert logs2["carry"].shape == (3,)
