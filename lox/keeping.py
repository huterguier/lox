import functools
from typing import Callable, Hashable, Iterable

import jax
import jax._src.ad_checkpoint
import jax.core
import jax.extend.core
from jax.extend.core import ClosedJaxpr, Jaxpr

from lox.primitive import lox_p
from lox.utils import flatten, is_hashable, select_logs

AxisName = Hashable


def keep(
    fun: Callable,
    argnames: str | Iterable[str] | None = None,
    tags: str | Iterable[str] | None = None,
) -> Callable:
    """
    Removes all logging operations from the given function except the ones matching
    ``argnames``/``tags``, by manipulating its Jaxpr. This is the whitelist complement
    of :func:`lox.strip`: where ``strip(f, argnames=X)`` removes ``X``,
    ``keep(f, argnames=X)`` removes everything *except* ``X``.

    Args:
      fun: The function from which to remove non-matching logging operations.
      argnames: An optional iterable of log keys to keep. ``None`` means no restriction
        on this axis; an empty iterable keeps nothing. If ``tags`` is also given, only
        entries matching both are kept.
      tags: An optional iterable of tags to keep. ``None`` means no restriction on this
        axis; an empty iterable keeps nothing.
    Returns:
      A new function with non-matching logging operations removed.

    Examples:
        By default lox.keep is a no-op: with no filter, everything is kept.

        >>> def f(x):
        >>>     lox.log({"x": x})
        >>>     return x + 1.0
        >>> y, logs = lox.spool(lox.keep(f))(1.0)
        >>> print(logs)
        {"x": 1.0}

        It is possible to specify which logged arguments to keep by providing their names.
        In this example only the logged argument "x" is kept, while "y" is stripped.

        >>> def f(x):
        >>>     y = x + 1.0
        >>>     lox.log({"x": x, "y": y})
        >>>     return x + 1.0
        >>> y, logs = lox.spool(lox.keep(f, argnames=["x"]))(1.0)
        >>> print(logs)
        {"x": 1.0}

        It is also possible to keep logged arguments based on their tags.
        All logged arguments with any of the specified tags will be kept, the rest stripped.

        >>> def f(x):
        >>>     y = x + 1.0
        >>>     lox.log({"x": x}, tags=["input"])
        >>>     lox.log({"y": y}, tags=["output"])
        >>>     return x + 1.0
        >>> y, logs = lox.spool(lox.keep(f, tags=["input"]))(1.0)
        >>> print(logs)
        {"x": 1.0}
    """

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        args_flat, structure = jax.tree.flatten((args, kwargs))
        static_argnums = tuple(i for i, arg in enumerate(args_flat) if is_hashable(arg))
        closed_jaxpr, out_shape = jax.make_jaxpr(
            flatten(fun, structure),
            static_argnums=static_argnums,
            return_shape=True,
        )(*args_flat)
        new_jaxpr = keep_jaxpr(closed_jaxpr.jaxpr, argnames=argnames, tags=tags)
        closed_jaxpr = ClosedJaxpr(new_jaxpr, closed_jaxpr.consts)
        dynamic_args_flat = tuple(arg for arg in args_flat if not is_hashable(arg))
        out_flat = jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *dynamic_args_flat)
        out = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(out_shape), out_flat
        )
        return out

    return wrapped


def keep_jaxpr(
    jaxpr: Jaxpr,
    argnames: str | Iterable[str] | None = None,
    tags: str | Iterable[str] | None = None,
) -> Jaxpr:
    """Remove all logging operations from a Jaxpr except those matching argnames/tags.
    Returns a new Jaxpr (does not mutate the input)."""
    new_eqns = []
    for eqn in jaxpr.eqns:
        if eqn.primitive == lox_p:
            logs_in = jax.tree.unflatten(eqn.params["structure"], eqn.invars)
            logs_out = jax.tree.unflatten(eqn.params["structure"], eqn.outvars)
            to_keep = select_logs(logs_in, eqn.params["tags"], argnames, tags)
            logs_in = logs_in.filter(lambda k, _: k in to_keep)
            logs_out = logs_out.filter(lambda k, _: k in to_keep)
            new_invars, new_structure = jax.tree.flatten(logs_in)
            new_eqns.append(eqn.replace(
                invars=new_invars,
                outvars=jax.tree.leaves(logs_out),
                params={**eqn.params, "structure": new_structure},
            ))
        elif eqn.primitive == jax.extend.core.primitives.scan_p:
            c = eqn.params["jaxpr"]
            new_eqns.append(eqn.replace(params={**eqn.params,
                "jaxpr": ClosedJaxpr(keep_jaxpr(c.jaxpr, argnames, tags), c.consts),
            }))
        elif eqn.primitive == jax.extend.core.primitives.cond_p:
            new_eqns.append(eqn.replace(params={**eqn.params, "branches": tuple(
                ClosedJaxpr(keep_jaxpr(b.jaxpr, argnames, tags), b.consts)
                for b in eqn.params["branches"]
            )}))
        elif eqn.primitive == jax.extend.core.primitives.while_p:
            c, b = eqn.params["cond_jaxpr"], eqn.params["body_jaxpr"]
            new_eqns.append(eqn.replace(params={**eqn.params,
                "cond_jaxpr": ClosedJaxpr(keep_jaxpr(c.jaxpr, argnames, tags), c.consts),
                "body_jaxpr": ClosedJaxpr(keep_jaxpr(b.jaxpr, argnames, tags), b.consts),
            }))
        elif eqn.primitive == jax.extend.core.primitives.jit_p:
            c = eqn.params["jaxpr"]
            new_eqns.append(eqn.replace(params={**eqn.params,
                "jaxpr": ClosedJaxpr(keep_jaxpr(c.jaxpr, argnames, tags), c.consts),
            }))
        elif eqn.primitive == jax.extend.core.primitives.call_p:
            new_eqns.append(eqn.replace(params={**eqn.params,
                "call_jaxpr": keep_jaxpr(eqn.params["call_jaxpr"], argnames, tags),
            }))
        elif eqn.primitive == jax._src.ad_checkpoint.remat_p:
            new_eqns.append(eqn.replace(params={**eqn.params,
                "jaxpr": keep_jaxpr(eqn.params["jaxpr"], argnames, tags),
            }))
        else:
            new_eqns.append(eqn)
    return jaxpr.replace(eqns=new_eqns)
