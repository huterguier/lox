from functools import wraps
from typing import Any, Callable, Hashable, Iterable

import jax
import jax._src.ad_checkpoint
import jax.core
import jax.extend
from jax.extend.core import ClosedJaxpr, Jaxpr

from lox.logdict import logdict
from lox.primitive import lox_p
from lox.utils import flatten, is_hashable, select_logs

AxisName = Hashable


def tap(
    fun: Callable,
    callback: Callable[[logdict], None] | None = None,
    argnames: str | Iterable[str] | None = None,
    tags: str | Iterable[str] | None = None,
    prefix: str = "",
) -> Callable:
    """
    A function transformation that taps into the execution of a JAX function and prints the values of specified arguments. One can only ``tap`` into values that are logged with :func:`lox.log`.
    This transformation modifies the function to exectute a callback with the tapped values. By default this callback will display the values in the console.
    It can be used to debug and inspect the values of arguments during the execution of a JAX function. It is possible to specify which arguments to tap by providing their names. If no names are provided, all arguments will be tapped.
    It is possible to provide a custom callback function to be called with the tapped values.
    The callback function should accept a single argument, which is a `logdict` containing the tapped values.

      >>> def callback(logs: logdict):
      >>>   print(logs)
      >>> def f(x, y):
      >>>   x = lox.log({"x": x, "y": y})
      >>>   return x + y, x * y
      >>> x, y = 1.0, 2.0
      >>> y = lox.tap(f, argnames=["y"], callback=callback)(x, y)
      {"y": 2.0}

    Note that this transformation can introduce a significant overhead, especially if the function is called frequently or with large inputs. It is recommended to use this transformation only for debugging purposes and to remove it before any performance-critical execution.
    Use :func:`lox.spool` to log values that you want to tap into.

    Args:
      fun: The function you want to tap into.
      callback: A callback function to be called with the tapped values. If None, the default callback will be used to display the values.
      argnames: A string or iterable of strings specifying the names of the arguments to be printed. If None, all arguments will be tapped.
      tags: A string or iterable of tags to filter the logs. If None, all tags will be tapped.
      prefix (str): An optional prefix to add to the log keys.
    Returns:
      Callable: A wrapped function that executes the original function and prints the tapped values.
    """

    @wraps(fun)
    def wrapped(*args, **kwargs):
        args_flat, structure = jax.tree_util.tree_flatten((args, kwargs))
        static_argnums = tuple(i for i, arg in enumerate(args_flat) if is_hashable(arg))
        closed_jaxpr, out_shape = make_tapped_jaxpr(
            flatten(fun, structure),
            static_argnums=static_argnums,
            callback=callback,
            argnames=argnames,
            tags=tags,
            prefix=prefix,
        )(*args_flat)
        dynamic_args_flat = tuple(arg for arg in args_flat if not is_hashable(arg))
        out_structure = jax.tree_util.tree_structure(out_shape)
        out_flat = jax.core.eval_jaxpr(
            closed_jaxpr.jaxpr, closed_jaxpr.literals, *dynamic_args_flat
        )
        out = jax.tree_util.tree_unflatten(out_structure, out_flat)
        return out

    return wrapped


def make_tapped_jaxpr(
    fun: Callable,
    static_argnums: int | Iterable[int] = (),
    callback: Callable[[logdict], None] | None = None,
    argnames: str | Iterable[str] | None = None,
    tags: str | Iterable[str] | None = None,
    prefix: str = "",
) -> Callable[..., tuple[ClosedJaxpr, Any]]:
    """
    Creates a JAX function that returns a closed Jaxpr with the specified arguments tapped.

    Args:
        fun (Callable): The function to create a jaxpr for.
        static_argnums (int | Iterable[int]): The indices of static arguments.
        callback (Callable[[logdict], None] | None): A callback function to be called with the tapped values. If None, the default callback will be used to display the values.
        argnames (str | Iterable[str] | None): The names of the arguments to be tapped. If None, all arguments will be tapped.
        prefix (str): An optional prefix to add to the log keys.
    Returns:
        Callable[..., ClosedJaxpr | tuple[ClosedJaxpr, Any]]: A wrapped function that returns the jaxpr and logs.
    """
    def wrapped(*args, **kwargs):
        closed_jaxpr, out_shape = jax.make_jaxpr(
            fun,
            static_argnums=static_argnums,
            return_shape=True,
        )(*args, **kwargs)
        new_jaxpr, _ = tap_jaxpr(
            closed_jaxpr.jaxpr,
            argnames=argnames,
            tags=tags,
            callback=callback if callback is not None else print,
            prefix=prefix,
        )
        return ClosedJaxpr(new_jaxpr, closed_jaxpr.consts), out_shape

    return wrapped


def tap_jaxpr(
    jaxpr: Jaxpr,
    callback: Callable[[logdict], None],
    argnames: str | Iterable[str] | None = None,
    tags: str | Iterable[str] | None = None,
    prefix: str = "",
) -> tuple[Jaxpr, bool]:
    """
    Taps into a JAX Jaxpr and inserts callback equations for logged values.
    Returns a new Jaxpr (does not mutate the input) and whether it was modified.
    The modified flag is needed so that jit_p is only replaced with call_p when
    the inner jaxpr actually contains logging.

    Args:
        jaxpr (Jaxpr): The Jaxpr to be tapped.
        callback (Callable[[logdict], None]): A callback function to be called with the tapped values.
        argnames (str | Iterable[str] | None): An iterable of argument names to be tapped.
        tags (str | Iterable[str] | None): An optional list of tags to filter the logs.
        prefix (str): An optional prefix to add to the log keys.
    Returns:
        tuple[Jaxpr, bool]: The new jaxpr and whether it was modified.
    """

    def wrapped_callback(structure, *logs_flat):
        def _callback(*logs_flat):
            logs = jax.tree.unflatten(structure, logs_flat)
            if prefix:
                logs = logs.prefix(prefix)
            callback(logs)

        jax.debug.callback(_callback, *logs_flat)

    new_eqns = []
    modified = False

    for eqn in jaxpr.eqns:

        if eqn.primitive == lox_p:
            structure = eqn.params["structure"]
            logs = jax.tree.unflatten(structure, eqn.invars)
            logs = select_logs(logs, eqn.params["tags"], argnames, tags)
            logs_avals = jax.tree.map(lambda l: l.aval, logs)
            logs_avals_flat, structure_avals = jax.tree.flatten(logs_avals)
            if logs_avals:
                print_jaxpr = jax.make_jaxpr(
                    wrapped_callback,
                    static_argnums=(0),
                )(structure_avals, *logs_avals_flat)
                new_eqns.append(print_jaxpr.jaxpr.eqns[0].replace(
                    invars=jax.tree.leaves(logs),
                ))
                modified = True
            new_eqns.append(eqn)

        elif eqn.primitive == jax.extend.core.primitives.jit_p:
            c = eqn.params["jaxpr"]
            new_inner, m = tap_jaxpr(c.jaxpr, callback, argnames)
            if m:
                new_eqns.append(eqn.replace(
                    primitive=jax.extend.core.primitives.call_p,
                    params={"call_jaxpr": new_inner},
                ))
                modified = True
            else:
                new_eqns.append(eqn)

        elif eqn.primitive == jax.extend.core.primitives.scan_p:
            c = eqn.params["jaxpr"]
            new_inner, m = tap_jaxpr(c.jaxpr, callback, argnames)
            modified |= m
            new_eqns.append(eqn.replace(params={**eqn.params,
                "jaxpr": ClosedJaxpr(new_inner, c.consts),
            }))

        elif eqn.primitive == jax.extend.core.primitives.cond_p:
            new_branches = []
            for b in eqn.params["branches"]:
                new_b, m = tap_jaxpr(b.jaxpr, callback, argnames)
                modified |= m
                new_branches.append(ClosedJaxpr(new_b, b.consts))
            new_eqns.append(eqn.replace(params={**eqn.params,
                "branches": tuple(new_branches),
            }))

        elif eqn.primitive == jax.extend.core.primitives.while_p:
            c, b = eqn.params["cond_jaxpr"], eqn.params["body_jaxpr"]
            new_c, mc = tap_jaxpr(c.jaxpr, callback, argnames)
            new_b, mb = tap_jaxpr(b.jaxpr, callback, argnames)
            modified |= mc | mb
            new_eqns.append(eqn.replace(params={**eqn.params,
                "cond_jaxpr": ClosedJaxpr(new_c, c.consts),
                "body_jaxpr": ClosedJaxpr(new_b, b.consts),
            }))

        elif eqn.primitive == jax.extend.core.primitives.call_p:
            new_call, m = tap_jaxpr(eqn.params["call_jaxpr"], callback, argnames)
            modified |= m
            new_eqns.append(eqn.replace(params={**eqn.params,
                "call_jaxpr": new_call,
            }))

        elif eqn.primitive == jax._src.ad_checkpoint.remat_p:
            new_remat, m = tap_jaxpr(eqn.params["jaxpr"], callback, argnames)
            modified |= m
            new_eqns.append(eqn.replace(params={**eqn.params,
                "jaxpr": new_remat,
            }))

        else:
            new_eqns.append(eqn)

    return jaxpr.replace(eqns=new_eqns), modified
