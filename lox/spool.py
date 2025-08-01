from functools import wraps
from typing import Any, Callable, Hashable, Iterable, Sequence

import jax
import jax.core
import jax.extend
import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jax._src import source_info_util
from jax.core import ShapedArray
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Var

from lox.logdict import logdict
from lox.nolog import nolog_jaxpr
from lox.primitive import lox_p
from lox.util import flatten, is_hashable

AxisName = Hashable


def spool(fun: Callable, keep_logs=False) -> Callable:
    """
    Spools a function to extract logs generated during its execution.

    Args:
      fun (Callable): The function to be spooled.
      keep_logs (bool): Whether to keep logs in the jaxpr.
    Returns:
        Callable: A wrapped function that returns the spooled jaxpr and logs.
    """

    @wraps(fun)
    def wrapped(*args, **kwargs):
        args_flat, structure = jax.tree.flatten((args, kwargs))
        static_argnums = tuple(i for i, arg in enumerate(args_flat) if is_hashable(arg))
        closed_jaxpr, out_shape = make_spooled_jaxpr(
            flatten(fun, structure),
            static_argnums=static_argnums,
            return_shape=True,
            keep_logs=keep_logs,
        )(*args_flat)
        dynamic_args_flat = tuple(arg for arg in args_flat if not is_hashable(arg))
        out_structure = jax.tree.structure(out_shape)
        out_flat = jax.core.eval_jaxpr(
            closed_jaxpr.jaxpr, closed_jaxpr.literals, *dynamic_args_flat
        )
        out = jax.tree_util.tree_unflatten(out_structure, out_flat)
        return out

    return wrapped


def make_spooled_jaxpr(
    fun: Callable,
    static_argnums: int | Iterable[int] = (),
    axis_env: Sequence[tuple[AxisName, int]] | None = None,
    return_shape: bool = False,
    abstracted_axes: Any | None = None,
    keep_logs: bool = False,
) -> Callable[..., ClosedJaxpr | tuple[ClosedJaxpr, Any]]:
    """
    Creates a spooled jaxpr for the given function, extracting logs and their shapes.

    Args:
        fun (Callable): The function to create a jaxpr for.
        static_argnums (int | Iterable[int]): The indices of static arguments.
        axis_env (Sequence[tuple[AxisName, int]] | None): The axis environment for the jaxpr.
        return_shape (bool): Whether to return the shape of the output.
        abstracted_axes (Any | None): Abstracted axes for the jaxpr.
        keep_logs (bool): Whether to keep logs in the jaxpr.
    Returns:
        Callable[..., ClosedJaxpr | tuple[ClosedJaxpr, Any]]: A wrapped function that returns the jaxpr and logs.
    """

    def wrapped(*args, **kwargs):
        closed_jaxpr, out_shape = jax.make_jaxpr(
            fun,
            static_argnums=static_argnums,
            axis_env=axis_env,
            return_shape=True,
            abstracted_axes=abstracted_axes,
        )(*args, **kwargs)
        logs = spool_jaxpr(closed_jaxpr.jaxpr)
        logs_shape = jax.tree_util.tree_map(
            lambda v: ShapeDtypeStruct(v.aval.shape, v.aval.dtype), logs
        )
        if not keep_logs:
            nolog_jaxpr(closed_jaxpr.jaxpr)
        if return_shape:
            return closed_jaxpr, (out_shape, logs_shape)
        else:
            return closed_jaxpr

    return wrapped


def apply(f: Callable, jaxpr: Jaxpr, *invars: Any) -> Any:
    """
    Applies a function to the invars within the context of the jaxpr.

    Args:
        f (Callable): The function to apply.
        jaxpr (Jaxpr): The jaxpr context in which to apply the function.
        *invars: The input variables to the function.
    Returns:
        Any: The output variables after applying the function.
    """
    invars_avals = jax.tree.map(lambda invar: invar.aval, invars)
    closed_jaxpr_f, shape_f = jax.make_jaxpr(f, return_shape=True)(*invars_avals)
    structure_f = jax.tree.structure(shape_f)
    jaxpr_f = closed_jaxpr_f.jaxpr
    jaxpr.eqns.append(
        JaxprEqn(
            primitive=jax.extend.core.primitives.call_p,
            invars=jax.tree.leaves(invars),
            outvars=jaxpr_f.outvars,
            params={"call_jaxpr": jaxpr_f},
            source_info=source_info_util.current(),
            effects=(),
            ctx=jaxpr.eqns[0].ctx,
        )
    )
    outvars = jax.tree.unflatten(structure_f, jaxpr_f.outvars)
    return outvars


def spool_jaxpr(jaxpr: Jaxpr) -> logdict:
    """
    Spools the logs from a jaxpr, extracting logs and their shapes from each equation.
    Combines logs from nested equations in the order they will be executed.

    Args:
        jaxpr (Jaxpr): The jaxpr to spool.
    Returns:
        tuple[dict[str, Any], dict[str, Any]]: The logs and their shapes.
    """
    logs_eqns = []
    for i in range(len(jaxpr.eqns)):
        eqn = jaxpr.eqns[i]
        logs_eqn = None
        if eqn.primitive == lox_p:
            logs_eqn = spool_lox_p(jaxpr, eqn)
        elif eqn.primitive == jax.lax.scan_p:
            logs_eqn = spool_scan_p(jaxpr, eqn)
        elif eqn.primitive == jax.lax.cond_p:
            logs_eqn = spool_cond_p(jaxpr, eqn)
        elif eqn.primitive == jax.lax.while_p:
            logs_eqn = spool_while_p(jaxpr, eqn)
        elif eqn.primitive.name == "pjit":
            logs_eqn = spool_pjit_p(jaxpr, eqn)
        elif eqn.primitive == jax.extend.core.primitives.call_p:
            logs_eqn = spool_call_p(jaxpr, eqn)

        if logs_eqn:
            logs_eqns.append(logs_eqn)

    if logs_eqns:

        def combine(logs_eqns):
            """Combines the logs from multiple equations into a single logdict."""
            logs_jaxpr = logs_eqns[0]
            for logs_eqn in logs_eqns[1:]:
                logs_jaxpr += logs_eqn
            return logs_jaxpr

        logs_jaxpr = apply(combine, jaxpr, logs_eqns)
        jaxpr.outvars.extend(jax.tree_util.tree_leaves(logs_jaxpr))
    else:
        logs_jaxpr = logdict({})

    return logs_jaxpr


def spool_lox_p(jaxpr: Jaxpr, eqn: JaxprEqn) -> logdict:
    """
    Spools the logs from a lox_p primitive. The logs are extracted from the equation's parameters.

    Args:
        eqn (JaxprEqn): The equation representing the lox_p operation.
    Returns:
        tuple[dict[str, Any], dict[str, Any]]: The logs and their shapes.
    """
    del jaxpr
    logs_eqn = jax.tree.unflatten(eqn.params["structure"], eqn.invars)
    return logs_eqn


def spool_scan_p(jaxpr: Jaxpr, eqn: JaxprEqn) -> logdict:
    """
    Spools the logs from a scan_p primitive. The logs of the jaxpr are reshaped to have a static length,
    which is the length of the scan.
    """
    logs_jaxpr = spool_jaxpr(eqn.params["jaxpr"].jaxpr)
    logs_jaxpr_avals = jax.tree_util.tree_map(lambda l: l.aval, logs_jaxpr)

    logs_scan_avals = jax.tree_util.tree_map(
        lambda aval: ShapedArray((eqn.params["length"],) + aval.shape, aval.dtype),
        logs_jaxpr_avals,
    )
    logs_scan = jax.tree.map(lambda aval: Var(aval=aval), logs_scan_avals)
    eqn.outvars.extend(jax.tree.leaves(logs_scan))

    def unstack(logs_scan):
        """Unstacks the logs from the scan, to have a single leading dimension."""
        return jax.tree.map(
            lambda l: l.reshape((-1,) + l.shape[2:]),
            logs_scan,
        )

    logs_eqn = apply(unstack, jaxpr, logs_scan)

    return logs_eqn


def spool_cond_p(jaxpr: Jaxpr, eqn: JaxprEqn) -> logdict:
    """
    Spools the branches of a cond_p primitive. All branches must have the same log structure and shapes.

    Args:
        eqn (JaxprEqn): The equation representing the switch operation.
    Returns:
        logdict: The logs and their shapes for the branches.
    Raises:
        ValueError: If the branches do not have the same log structure or shapes.
    """
    del jaxpr
    branches = eqn.params["branches"]
    logs_branches = []

    for branch in branches:
        logs_branch = spool_jaxpr(branch.jaxpr)
        logs_branches.append(logs_branch)

    logs_eqn = jax.tree.map(
        lambda l: Var(aval=ShapedArray(l.aval.shape, l.aval.dtype)), logs_branches[0]
    )
    eqn.outvars.extend(jax.tree.leaves(logs_eqn))

    return logs_eqn


def spool_while_p(jaxpr: Jaxpr, eqn: JaxprEqn) -> logdict:
    """
    Spools the inner jaxpr of a while_p primitive. If the jaxpr contains any logging operations,
        it raises an error since while loops have non-static lengths.

    Args:
        eqn (JaxprEqn): The equation representing the while loop.
    Returns:
        logdict: An empty logdict, as spooling is not supported for while loops.
    Raises:
        ValueError: If the jaxpr contains any logging operations, since while loops have non-static lengths.
    """
    del jaxpr
    logs_cond = spool_jaxpr(eqn.params["cond_jaxpr"].jaxpr)
    logs_body = spool_jaxpr(eqn.params["body_jaxpr"].jaxpr)
    if logs_cond or logs_body:
        raise ValueError(
            "Spooling for while loops is not supported due to non-static length."
        )
    return logdict({})


def spool_pjit_p(jaxpr: Jaxpr, eqn: JaxprEqn) -> logdict:
    """
    Spools the jaxpr of a pjit primitive. As spooling the function would trigger recompilation,
    the wrapping pjit is removed if the jaxpr contains any lox_p primitives.

    Args:
        eqn (JaxprEqn): The equation representing the pjit operation.
    Returns:
        tuple[dict[str, Any], dict[str, Any]]: The logs and their shapes.
    """
    del jaxpr
    logs_jaxpr = spool_jaxpr(eqn.params["jaxpr"].jaxpr)
    if logs_jaxpr:
        logs_eqn = jax.tree.map(
            lambda l: Var(aval=ShapedArray(l.aval.shape, l.aval.dtype)), logs_jaxpr
        )
        eqn.outvars.extend(jax.tree.leaves(logs_eqn))
        eqn.primitive = jax.extend.core.primitives.call_p
        eqn.params = {"call_jaxpr": eqn.params["jaxpr"].jaxpr}
        return logs_eqn
    else:
        return logdict({})


def spool_call_p(jaxpr: Jaxpr, eqn: JaxprEqn) -> logdict:
    """
    Spools the jaxpr of a call_p primitive. This is used to handle the case where a function is called
    within a jaxpr, allowing us to track logs from the called function.

    Args:
        eqn (JaxprEqn): The equation representing the call operation.
    Returns:
        tuple[dict[str, Any], dict[str, Any]]: The logs and their shapes.
    """
    del jaxpr
    logs_call_jaxpr = spool_jaxpr(eqn.params["call_jaxpr"])
    logs_eqn = jax.tree.map(
        lambda l: Var(aval=ShapedArray(l.aval.shape, l.aval.dtype)), logs_call_jaxpr
    )
    eqn.outvars.extend(jax.tree_util.tree_leaves(logs_eqn))
    return logs_eqn


from .primitive import log


def f(xs):
    def step(carry, x):
        log({"x": x})
        x = jax.lax.cond(x > 0, true_fun, false_fun, x)
        return carry + x, carry + x

    def true_fun(x):
        log({"x": x - 1})
        return x + 1

    def false_fun(x):
        log({"x": x + 1})
        return x - 1

    carry = 0
    x = jax.lax.scan(step, carry, xs)
    log({"x": carry})
    return x


x = jnp.array([1, 2, 3])
y, logs = spool(f)(x)
print(logs)
