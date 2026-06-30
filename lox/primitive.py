from typing import Any, Iterable

import jax
import jax.numpy as jnp
from jax._src.debugging import DebugEffect
from jax.extend import core
from jax.interpreters import ad, batching, mlir

from lox.logdict import logdict

lox_p = core.Primitive("lox")
lox_p.multiple_results = True


def log(data: dict[str, Any], tags: Iterable[str] = ()) -> logdict:
    """
    Fundamental logging primitive for Lox.

    Args:
        data: A dictionary containing the data to be logged.
        tags: An iterable of strings representing tags associated with the log.

    Returns:
        logdict: A logdict object containing the logged data.

    Examples:
        By default lox.log complies with the pure functional programming paradigm of JAX,
        meaning it does not have side effects and does not mutate the state.

        >>> def f_log(x):
        >>>     lox.log({"x": x})
        >>>     return x + 1.0
        >>> f_log(1.0)
        2.0

        In order to actually retrieve or use the logged data, we need to apply a function
        transformation such as :attr:`spool` or :attr:`tap`.

    """
    data_logdict = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), data)
    logs = logdict(data_logdict)
    logs_flat, structure = jax.tree_util.tree_flatten(logs)
    _ = lox_p.bind(*logs_flat, tags=tuple(tags), structure=structure)
    return jax.tree_util.tree_unflatten(structure, logs_flat)


@lox_p.def_impl
def lox_impl(*logs_flat, tags, structure):
    del structure, tags
    return logs_flat


@lox_p.def_effectful_abstract_eval
def lox_abstract_eval(*logs_flat, tags, structure):
    del structure, tags
    return list(logs_flat), {DebugEffect()}


def lox_lowering(*logs_flat, tags, structure):
    del structure, tags
    return logs_flat


mlir.register_lowering(lox_p, mlir.lower_fun(lox_lowering, multiple_results=True))


def lox_batch(vector_arg_values, batch_axes, tags, structure):
    outs = lox_p.bind(*vector_arg_values, tags=tags, structure=structure)
    return outs, batch_axes


batching.primitive_batchers[lox_p] = lox_batch


def lox_jvp(arg_values, arg_tangents, tags, structure):
    lox_p.bind(*arg_values, tags=tags, structure=structure)
    return arg_values, arg_tangents


ad.primitive_jvps[lox_p] = lox_jvp


def lox_p_transpose(ct, x):
    del ct, x
    raise ValueError("Transpose doesn't support logging")


ad.primitive_transposes[lox_p] = lox_p_transpose
