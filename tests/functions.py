import jax
import jax.numpy as jnp

import lox


def f_id(x):
    lox.log({"x": x})
    return x


def f_add(x):
    z = x + x
    lox.log({"x": x, "z": z})
    return z


def f_scan(x):
    def step(carry, x):
        carry = carry + x.mean()
        lox.log({"carry": carry, "x": x})
        return carry, carry

    return jax.lax.scan(step, 0, x)


def f_call(x):
    def g(x):
        lox.log({"x": x})
        return x * 2

    return g(x) + 1


def f_jit(x):
    @jax.jit
    def g(x):
        lox.log({"x": x})
        return x * 3

    return g(x) + 1


def f_cond(x):
    def true_fun(x):
        x = x + 1
        lox.log({"branch": True, "x": x})
        return x

    def false_fun(x):
        x = x - 1
        lox.log({"branch": False, "x": x})
        return x

    cond = x.ravel()[0] > 0
    return jax.lax.cond(cond, true_fun, false_fun, x)


def f_grad(x):
    def func(x):
        lox.log({"x": x})
        return x.mean()

    grad_func = jax.grad(func)
    return grad_func(x)


def f_remat(x):
    @jax.remat
    def g(x):
        lox.log({"x": x})
        return x * 2

    return g(x) + 1


def f_while(x):
    def cond(state):
        i, _ = state
        return i < x.shape[0]

    def body(state):
        i, carry = state
        lox.log({"carry": carry})
        return i + 1, carry + x[i].mean()

    _, result = jax.lax.while_loop(cond, body, (jnp.array(0), jnp.array(0.0)))
    return result
