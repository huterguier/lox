import functools
from typing import Callable, Hashable, Iterable

import jax
import jax._src.ad_checkpoint
import jax.core
import jax.extend.core
from jax.extend.core import Jaxpr

from lox.logdict import logdict
from lox.primitive import lox_p

AxisName = Hashable


def strip(
    fun: Callable,
    argnames: Iterable[str] | None = None,
    tags: Iterable[str] | None = None,
) -> Callable:
    """
    Strips all logging operations from the given function by manipulating its Jaxpr.
    This is useful when you remove logging statements from parts of you code or want to quickly disable logging for performance reasons.

    Args:
      fun: The function from which to strip logging operations.
    Returns:
      A new function with logging operations removed.

    Examples:
        By default lox.strip removes all logging operations from a function.

        >>> def f(x):
        >>>     lox.log({"x": x})
        >>>     return x + 1.0
        >>> y, logs = lox.spool(lox.strip(f))(1.0)
        >>> print(logs)
        {}

        It is possible to specify which logged arguments to strip by providing their names. In this example we only strip the logged argument "x", while "y" is kept.

        >>> def f(x):
        >>>     y = x + 1.0
        >>>     lox.log({"x": x, "y": y})
        >>>     return x + 1.0
        >>> y, logs = lox.spool(lox.strip(f, argnames=["x"]))(1.0)
        >>> print(logs)
        {"y": 2.0}

        It is also possible to strip logged arguments based on their tags.
        All logged arguments with any of the specified tags will be stripped from the function.

        >>> def f(x):
        >>>     y = x + 1.0
        >>>     lox.log({"x": x}, tags=["input"])
        >>>     lox.log({"y": y}, tags=["output"])
        >>>     return x + 1.0
        >>> y, logs = lox.spool(lox.strip(f, tags=["input"]))(1.0)
        >>> print(logs)
        {"y": 2.0}
    """

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        closed_jaxpr, out_shape = jax.make_jaxpr(fun, return_shape=True)(
            *args, **kwargs
        )
        strip_jaxpr(closed_jaxpr.jaxpr, argnames=argnames, tags=tags)
        out_flat = jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
        out = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(out_shape), out_flat
        )
        return out

    return wrapped


def strip_jaxpr(
    jaxpr: Jaxpr,
    argnames: Iterable[str] | None = None,
    tags: Iterable[str] | None = None,
):
    """Remove all logging operations from a Jaxpr."""
    for eqn in jaxpr.eqns:
        if eqn.primitive == lox_p:
            logs_in = jax.tree.unflatten(eqn.params["structure"], eqn.invars)
            logs_out = jax.tree.unflatten(eqn.params["structure"], eqn.outvars)
            if tags is not None and any(tag in tags for tag in eqn.params["tags"]):
                logs_in = logdict({})
                logs_out = logdict({})
            elif argnames:
                logs_in = logs_in.filter(lambda k, _: k not in argnames)
                logs_out = logs_out.filter(lambda k, _: k not in argnames)
            eqn.invars, eqn.params["structure"] = jax.tree.flatten(logs_in)
            eqn.outvars = jax.tree.leaves(logs_out)
        elif eqn.primitive == jax.extend.core.primitives.scan_p:
            strip_jaxpr(eqn.params["jaxpr"])
        elif eqn.primitive == jax.extend.core.primitives.cond_p:
            for branch in eqn.params["branches"]:
                strip_jaxpr(branch.jaxpr)
        elif eqn.primitive == jax.extend.core.primitives.while_p:
            strip_jaxpr(eqn.params["body_jaxpr"])
            strip_jaxpr(eqn.params["cond_jaxpr"])
        elif eqn.primitive == jax.extend.core.primitives.jit_p:
            strip_jaxpr(eqn.params["jaxpr"])
        elif eqn.primitive == jax.extend.core.primitives.call_p:
            strip_jaxpr(eqn.params["call_jaxpr"])
        elif eqn.primitive == jax._src.ad_checkpoint.remat_p:
            strip_jaxpr(eqn.params["jaxpr"])
