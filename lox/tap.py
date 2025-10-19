from functools import wraps
from typing import Any, Callable, Hashable, Iterable, Sequence

import jax
import jax.core
import jax.extend
from jax.extend.core import ClosedJaxpr, Jaxpr

from lox.logdict import logdict
from lox.primitive import lox_p
from lox.utils import flatten, is_hashable

AxisName = Hashable


def tap(
    fun: Callable,
    callback: Callable[[logdict], None] | None = None,
    argnames: str | Iterable[str] | None = None,
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
            return_shape=True,
            callback=callback,
            argnames=argnames,
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
    axis_env: Sequence[tuple[AxisName, int]] | None = None,
    return_shape: bool = False,
    abstracted_axes: Any | None = None,
    callback: Callable[[logdict], None] | None = None,
    argnames: str | Iterable[str] | None = None,
) -> Callable[..., tuple[ClosedJaxpr, Any]]:
    """
    Creates a JAX function that returns a closed Jaxpr with the specified arguments tapped.

    Args:
        fun (Callable): The function to create a jaxpr for.
        static_argnums (int | Iterable[int]): The indices of static arguments.
        axis_env (Sequence[tuple[AxisName, int]] | None): The axis environment for the jaxpr.
        return_shape (bool): Whether to return the shape of the output.
        abstracted_axes (Any | None): Abstracted axes for the jaxpr.
        callback (Callable[[logdict], None] | None): A callback function to be called with the tapped values. If None, the default callback will be used to display the values.
        argnames (str | Iterable[str] | None): The names of the arguments to be tapped. If None, all arguments will be tapped.
    Returns:
        Callable[..., ClosedJaxpr | tuple[ClosedJaxpr, Any]]: A wrapped function that returns the jaxpr and logs.
    """
    if argnames is str:
        argnames = (argnames,)

    def wrapped(*args, **kwargs):
        closed_jaxpr, out_shape = jax.make_jaxpr(
            fun,
            static_argnums=static_argnums,
            axis_env=axis_env,
            return_shape=True,
            abstracted_axes=abstracted_axes,
        )(*args, **kwargs)
        _ = tap_jaxpr(
            closed_jaxpr.jaxpr,
            argnames=argnames,
            callback=callback if callback is not None else print,
        )
        return closed_jaxpr, out_shape

    return wrapped


def tap_jaxpr(
    jaxpr: Jaxpr,
    callback: Callable[[logdict], None],
    argnames: Iterable[str] | None = None,
):
    """
    Taps into a JAX Jaxpr and prints the values of specified arguments. Recurisvely traverses the Jaxpr to find and tap into `lox_p` primitives.

    Args:
        jaxpr (Jaxpr): The Jaxpr to be tapped.
        callback (Callable[[logdict], None]): A callback function to be called with the tapped values. It should accept a single argument, which is a `logdict` containing the tapped values.
        argnames (Iterable[str] | None): An iterable of argument names to be tapped. If None, all arguments will be tapped.
    Returns:
      bool: True if the Jaxpr was modified, False otherwise.
    """

    def wrapped_callback(structure, *logs_flat):
        def _callback(*logs_flat):
            logs = jax.tree.unflatten(structure, logs_flat)
            callback(logs)

        jax.debug.callback(_callback, *logs_flat)

    i = 0
    modified = False

    while i < len(jaxpr.eqns):
        eqn = jaxpr.eqns[i]

        if eqn.primitive == lox_p:
            structure = eqn.params["structure"]
            logs = jax.tree.unflatten(structure, eqn.invars)
            if argnames is None and eqn.params["explicit"]:
                logs = logdict({})
            elif argnames is not None:
                logs = logs.filter(lambda k, _: k in argnames)
            logs_avals = jax.tree.map(lambda l: l.aval, logs)
            logs_avals_flat, structure_avals = jax.tree.flatten(logs_avals)
            if logs_avals:
                print_jaxpr = jax.make_jaxpr(
                    wrapped_callback,
                    static_argnums=(0),
                )(
                    structure_avals,
                    *logs_avals_flat,
                )
                jaxpr.eqns.insert(i, print_jaxpr.jaxpr.eqns[0])
                jaxpr.eqns[i].invars = jax.tree.leaves(logs)
                i += 1
                modified = True

        elif eqn.primitive == jax.lax.scan_p:
            modified |= tap_jaxpr(eqn.params["jaxpr"].jaxpr, callback, argnames)

        elif eqn.primitive == jax.lax.cond_p:
            branches = eqn.params["branches"]
            for branch in branches:
                modified |= tap_jaxpr(branch.jaxpr, callback, argnames)

        elif eqn.primitive == jax.lax.while_p:
            modified |= tap_jaxpr(eqn.params["cond_jaxpr"].jaxpr, callback, argnames)
            modified |= tap_jaxpr(eqn.params["body_jaxpr"].jaxpr, callback, argnames)

        elif eqn.primitive.name == "pjit":
            modified_call_jaxpr = tap_jaxpr(eqn.params["jaxpr"], callback, argnames)
            if modified_call_jaxpr:
                eqn.primitive = jax.extend.core.primitives.call_p
                eqn.params = {"call_jaxpr": eqn.params["jaxpr"].jaxpr}
                modified = True

        elif eqn.primitive == jax.extend.core.primitives.call_p:
            modified |= tap_jaxpr(eqn.params["call_jaxpr"], callback, argnames)

        i += 1
    return modified
