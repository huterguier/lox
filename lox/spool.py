import jax
import jax.core
import jax.extend
import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jax.core import ShapedArray, AxisName
from jax.extend.core import Var, ClosedJaxpr, Jaxpr, JaxprEqn
from jax._src import source_info_util
from typing import Any, Iterable, Sequence, Callable
from lox.primitive import lox_p
from lox.nolog import nolog_jaxpr
from lox.util import is_hashable, flatten
from lox.logdict import logdict
from functools import wraps


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
    out_flat = jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *dynamic_args_flat)
    out = jax.tree_util.tree_unflatten(out_structure, out_flat)
    print("out", out)
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
    print(logs)
    logs_shape = jax.tree_util.tree_map(lambda v: ShapeDtypeStruct(v.aval.shape, v.aval.dtype), logs)
    print(logs_shape)
    if not keep_logs:
      nolog_jaxpr(closed_jaxpr.jaxpr)
    if return_shape:
      return closed_jaxpr, (out_shape, logs_shape)
    else:
      return closed_jaxpr
  return wrapped


def spool_jaxpr(jaxpr: Jaxpr) -> logdict:
  """
  Spools the logs from a jaxpr, extracting logs and their shapes from each equation.
  Combines logs from nested equations in the order they will be executed.

  Args:
      jaxpr (Jaxpr): The jaxpr to spool.
  Returns:
      tuple[dict[str, Any], dict[str, Any]]: The logs and their shapes.
  """

  def combine(logs_eqns):
    logs_jaxpr = logs_eqns[0]
    for logs_eqn in logs_eqns[1:]:
      logs_jaxpr += logs_eqn
    return logs_jaxpr

  logs_eqns = []
  for eqn in jaxpr.eqns:
    logs_eqn = None
    if eqn.primitive == lox_p:
      logs_eqn = spool_lox_p(eqn)
    elif eqn.primitive == jax.lax.scan_p:
      logs_eqn = spool_scan_p(jaxpr, eqn)
    elif eqn.primitive == jax.lax.cond_p:
      logs_eqn = spool_cond_p(eqn)
    elif eqn.primitive == jax.lax.while_p:
      logs_eqn = spool_while_p(eqn)
    elif eqn.primitive.name == "pjit":
      logs_eqn = spool_pjit_p(eqn)
    elif eqn.primitive == jax.extend.core.primitives.call_p:
      logs_eqn = spool_call_p(eqn)

    if logs_eqn:
      logs_eqns.append(logs_eqn)


  if logs_eqns:
    logs_avals = jax.tree_util.tree_map(lambda v: v.aval, logs_eqns)
    closed_jaxpr_combine, shape_combine = jax.make_jaxpr(combine, return_shape=True)(logs_avals)
    jaxpr_combine = closed_jaxpr_combine.jaxpr

    structure_combine = jax.tree_util.tree_structure(shape_combine)

    jaxpr.eqns.append(JaxprEqn(
        primitive=jax.extend.core.primitives.call_p,
        invars=jax.tree_util.tree_leaves(logs_eqns),
        outvars=jaxpr_combine.outvars,
        params={"call_jaxpr": jaxpr_combine},
        source_info=source_info_util.current(),
        effects=(),
        ctx=jaxpr.eqns[0].ctx,
    ))
    logs_jaxpr = jax.tree_util.tree_unflatten(structure_combine, jaxpr_combine.outvars)
    jaxpr.outvars.extend(jax.tree_util.tree_leaves(logs_jaxpr))
  else:
    logs_jaxpr = logdict({})

  return logs_jaxpr


def spool_lox_p(eqn: JaxprEqn) -> tuple[dict[str, Any], dict[str, Any]]:
  """
  Spools the logs from a lox_p primitive. The logs are extracted from the equation's parameters.

  Args:
      eqn (JaxprEqn): The equation representing the lox_p operation.
  Returns:
      tuple[dict[str, Any], dict[str, Any]]: The logs and their shapes.
  """
  logs_eqn = jax.tree.unflatten(eqn.params["structure"], eqn.invars)
  return logs_eqn


def spool_scan_p(jaxpr: Jaxpr, eqn: JaxprEqn) -> logdict:
  """
  Spools the logs from a scan_p primitive. The logs of the jaxpr are reshaped to have a static length,
  which is the length of the scan.

  Args:
      eqn (JaxprEqn): The equation representing the scan operation.
  Returns:
      tuple[dict[str, Any], dict[str, Any]]: The logs and their shapes.
  """
  logs_jaxpr = spool_jaxpr(eqn.params["jaxpr"].jaxpr)
  logs_jaxpr_avals = jax.tree_util.tree_map(lambda l: l.aval, logs_jaxpr)

  logs_scan_avals = jax.tree_util.tree_map(
      lambda aval: ShapedArray((eqn.params["length"],) + aval.shape, aval.dtype),
      logs_jaxpr_avals
  )
  logs_scan = jax.tree.map(lambda aval: Var("", aval=aval), logs_scan_avals)
  eqn.outvars.extend(jax.tree_util.tree_leaves(logs_scan))

  jaxpr_squeeze, logs_eqn_shape = jax.make_jaxpr(
      lambda logs: jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=1), logs),
      return_shape=True,
  )(logs_scan_avals)
  jaxpr.eqns.append(JaxprEqn(
      primitive=jax.extend.core.primitives.call_p,
      invars=logs_scan,
      outvars=jaxpr_squeeze.jaxpr.outvars,
      params={"call_jaxpr": jaxpr_squeeze.jaxpr},
      source_info=source_info_util.current(),
      effects=(),
      ctx=eqn.ctx,
  ))
  logs_eqn = jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(logs_eqn_shape),
      jaxpr_squeeze.jaxpr.outvars
  )
  return logs_eqn


def spool_cond_p(eqn: JaxprEqn) -> tuple[dict[str, Any], dict[str, Any]]:
  """
  Spools the branches of a cond_p primitive. All branches must have the same log structure and shapes.

  Args:
      eqn (JaxprEqn): The equation representing the switch operation.
  Returns:
      tuple[dict[str, Any], dict[str, Any]]: The logs and their shapes.
  Raises:
      ValueError: If the branches do not have the same log structure or shapes.
  """
  branches = eqn.params["branches"]
  logs_branches = []

  for branch in branches:
    branch_logs_shape, branch_log_shapes = spool_jaxpr(branch.jaxpr)
    branches_logs.append(branch_logs_shape)
    branches_log_shapes.append(branch_log_shapes)

  if not all(branches_logs[0] == branch_logs for branch_logs in branches_logs):
    raise ValueError("All branches must have the same log structure.")
  if not all(branches_log_shapes[0] == shape for shape in branches_log_shapes):
    raise ValueError("All branches must have the same log shapes.")

  logs_eqn = jax.tree.map(lambda s: Var("", aval=ShapedArray(s.shape, s.dtype)), branches_logs[0])
  eqn_log_shapes = branches_log_shapes[0]
  eqn.outvars.extend(jax.tree.leaves(logs_eqn))

  return logs_eqn, eqn_log_shapes


def spool_while_p(eqn: JaxprEqn) -> tuple[dict[str, Any], dict[str, Any]]:
  """
  Spools the inner jaxpr of a while_p primitive. If the jaxpr contains any logging operations,
      it raises an error since while loops have non-static lengths.

  Args:
      eqn (JaxprEqn): The equation representing the while loop.
  Returns:
      tuple[dict[str, Any], dict[str, Any]]: The logs and their shapes.
  Raises:
      ValueError: If the jaxpr contains any logging operations, since while loops have non-static lengths.
  """
  cond_logs_shape, cond_log_shapes = spool_jaxpr(eqn.params["cond_jaxpr"].jaxpr)
  body_logs_shape, body_log_shapes = spool_jaxpr(eqn.params["body_jaxpr"].jaxpr)
  if cond_logs_shape or cond_log_shapes or body_logs_shape or body_log_shapes:
    raise ValueError("Spooling for while loops is not supported due to non-static length.")
  return {}, {}


def spool_pjit_p(eqn: JaxprEqn) -> tuple[dict[str, Any], dict[str, Any]]:
  """
  Spools the jaxpr of a pjit primitive. As spooling the function would trigger recompilation,
  the wrapping pjit is removed if the jaxpr contains any lox_p primitives.

  Args:
      eqn (JaxprEqn): The equation representing the pjit operation.
  Returns:
      tuple[dict[str, Any], dict[str, Any]]: The logs and their shapes.
  """
  jaxpr_logs_shape, jaxpr_log_shapes = spool_jaxpr(eqn.params["jaxpr"].jaxpr)
  if jaxpr_logs_shape or jaxpr_log_shapes:
    logs_eqn = jax.tree.map(lambda s: Var("", aval=ShapedArray(s.shape, s.dtype)), jaxpr_logs_shape)
    eqn_log_shapes = jaxpr_log_shapes

    eqn.outvars.extend(jax.tree.leaves(logs_eqn))
    eqn.primitive = jax.extend.core.primitives.call_p
    eqn.params = {"call_jaxpr": eqn.params["jaxpr"].jaxpr}
    return logs_eqn, eqn_log_shapes
  else:
    return {}, {}


def spool_call_p(eqn: JaxprEqn) -> tuple[dict[str, Any], dict[str, Any]]:
  """
  Spools the jaxpr of a call_p primitive. This is used to handle the case where a function is called
  within a jaxpr, allowing us to track logs from the called function.

  Args:
      eqn (JaxprEqn): The equation representing the call operation.
  Returns:
      tuple[dict[str, Any], dict[str, Any]]: The logs and their shapes.
  """
  logs_jaxpr = spool_jaxpr(eqn.params["call_jaxpr"])
  logs_eqn = jax.tree.map(lambda v: Var("", aval=ShapedArray(v.aval.shape, v.aval.dtype)), logs_jaxpr)
  eqn.outvars.extend(jax.tree_util.tree_leaves(logs_eqn))
  return logs_eqn
