import jax

import lox


def f_id(x):
    lox.log({"x": x})
    return x


def f_id_spooled(x):
    return x, lox.log({"x": x})


def f_add(x):
    z = x + x
    lox.log({"x": x, "z": z})
    return z


def f_add_spooled(x):
    z = x + x
    return z, lox.log({"x": x, "z": z})


def f_scan(x):
    def step(carry, x):
        carry = carry + x.mean()
        lox.log({"carry": carry, "x": x})
        return carry, carry

    return jax.lax.scan(step, 0, x)


def f_scan_spooled(x):
    def step(carry, x):
        carry = carry + x.mean()
        return carry, (carry, {"carry": carry, "x": x})

    carry, (ys, data) = jax.lax.scan(step, 0, x)
    return (carry, ys), lox.log(data)


def f_call(x):
    def g(x):
        lox.log({"x": x})
        return x * 2

    return g(x) + 1


def f_call_spooled(x):
    def g(x):
        y = x * 2
        return y, lox.log({"x": x})

    y, logs = g(x)
    return y + 1, logs


def f_jit(x):
    @jax.jit
    def g(x):
        lox.log({"x": x})
        return x * 3

    return g(x) + 1


def f_jit_spooled(x):
    @jax.jit
    def g(x):
        y = x * 3
        return y, lox.log({"x": x})

    y, logs = g(x)
    return y + 1, logs


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


def f_cond_spooled(x):
    def true_fun(x):
        x = x + 1
        return x, lox.log({"branch": True, "x": x})

    def false_fun(x):
        x = x - 1
        return x, lox.log({"branch": False, "x": x})

    cond = x.ravel()[0] > 0
    y, logs = jax.lax.cond(cond, true_fun, false_fun, x)
    return y, logs


def f_grad(x):
    def func(x):
        lox.log({"x": x})
        return x.mean()

    grad_func = jax.grad(func)
    return grad_func(x)


def f_grad_spooled(x):
    def func(x):
        y = x.mean()
        return y, lox.log({"x": x})

    def wrapped_func(x):
        y, log = func(x)
        return y, log

    grad_func = jax.grad(wrapped_func, has_aux=True)

    (grad_value, log) = grad_func(x)
    return grad_value, log
