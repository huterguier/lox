import jax
import jax.core
import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jax.core import ShapedArray, AxisName
from jax.extend.core import Var, ClosedJaxpr, Jaxpr, JaxprEqn
from jax._src import source_info_util
from typing import Any, Iterable, Sequence, Callable

from jax.tree_util import PyTreeDef 
from lox.primitive import lox_p
from lox.nolog import nolog_jaxpr

from functools import wraps


def is_hashable(arg):
  try:
    hash(arg)
    return True
  except TypeError:
    return False


def spool(fun: Callable, keep_logs=False) -> Callable:
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
    return out
  return wrapped


def flatten(fun: Callable, structure: PyTreeDef) -> Callable:
  """
    Transforms a function to accept a single flat argument list.

    Args:
        fun (Callable): The function to be transformed.
    Returns:
        Callable: A new function that accepts a single flat argument list.
  """
  def wrapped(*args_flat):
    args, kwargs = jax.tree.unflatten(structure, args_flat)
    out = fun(*args, **kwargs)
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
  
  def wrapped(*args, **kwargs):
    closed_jaxpr, out_shape = jax.make_jaxpr(
        fun,
        static_argnums=static_argnums,
        axis_env=axis_env,
        return_shape=True,
        abstracted_axes=abstracted_axes,
    )(*args, **kwargs)
    logs_shape, log_shapes = spool_jaxpr(closed_jaxpr.jaxpr)
    if not keep_logs:
      nolog_jaxpr(closed_jaxpr.jaxpr)
    if return_shape:
      return closed_jaxpr, (out_shape, logs_shape)
    else:
      return closed_jaxpr
  return wrapped


def spool_jaxpr(jaxpr: Jaxpr) -> tuple[dict[str, Any], dict[str, Any]]:

  logs: dict[str, Any] = {}
  log_shapes: dict[str, Any] = {}

  def add_logs(eqn_logs: dict[str, Any]):
    for k, v in eqn_logs.items():
      if k in logs:
        logs[k].append(v)
      else:
        logs[k] = [v]

  def add_log_shapes(eqn_log_shapes):
    for k, v in eqn_logs.items():
      if k in log_shapes:
        log_structures = jax.tree.structure(log_shapes[k])
        eqn_log_structures = jax.tree.structure(eqn_log_shapes[k])
        if log_structures != eqn_log_structures:
          raise ValueError(f"Log structure mismatch for {k}: {log_structures} vs {eqn_log_structures}")
        if not jax.tree.all(jax.tree.map(lambda x, y: x.shape == y.shape, log_shapes[k], eqn_log_shapes[k])):
          raise ValueError(f"Log shape mismatch for {k}: {log_shapes[k]} vs {eqn_log_shapes[k]}")
        if not jax.tree.all(jax.tree.map(lambda x, y: x.dtype == y.dtype, log_shapes[k], eqn_log_shapes[k])):
          raise ValueError(f"Log dtype mismatch for {k}: {log_shapes[k]} vs {eqn_log_shapes[k]}")
      else:
        log_shapes[k] = eqn_log_shapes[k]


  for eqn in jaxpr.eqns:
    eqn_logs, eqn_log_shapes = {}, {}
    if eqn.primitive == lox_p:
      eqn_logs, eqn_log_shapes = spool_lox_p(eqn)
    elif eqn.primitive == jax.lax.scan_p:
      eqn_logs, eqn_log_shapes = spool_scan_p(eqn)
    elif eqn.primitive == jax.lax.cond_p:
      eqn_logs, eqn_log_shapes = spool_cond_p(eqn)
    elif eqn.primitive == jax.lax.while_p:
      eqn_logs, eqn_log_shapes = spool_while_p(eqn)
    elif eqn.primitive.name == "pjit":
      eqn_logs, eqn_log_shapes = spool_pjit_p(eqn)
    elif eqn.primitive == jax.extend.core.primitives.call_p:
      eqn_logs, eqn_log_shapes = spool_call_p(eqn)

    add_logs(eqn_logs)
    add_log_shapes(eqn_log_shapes)

  def combine(logs):
    for k, vals in logs.items():
      reshaped_vals = []
      for val in vals:
        reshaped_val = jax.tree.map(lambda v, shape: jnp.reshape(v, (-1,) + shape.shape), val, log_shapes[k])
        reshaped_vals.append(reshaped_val)

      if len(reshaped_vals) == 1:
        logs[k] = reshaped_vals[0]
      else:
        logs[k] = jnp.concatenate(reshaped_vals, axis=0)
    return logs

  if logs:
    logs_avals = jax.tree.map(lambda v: v.aval, logs)
    combine_jaxpr, logs_shape = jax.make_jaxpr(combine, return_shape=True)(logs_avals)
    jaxpr.eqns.append(JaxprEqn(
        primitive=jax.extend.core.primitives.call_p,
        invars=jax.tree.leaves(logs),
        outvars=combine_jaxpr.jaxpr.outvars,
        params={"call_jaxpr": combine_jaxpr.jaxpr},
        source_info=source_info_util.current(),
        effects=(),
        ctx=jaxpr.eqns[0].ctx,
    ))
    jaxpr.outvars.extend(combine_jaxpr.jaxpr.outvars)
  else:
    logs_shape = {}

  return logs_shape, log_shapes


def spool_lox_p(eqn: JaxprEqn) -> tuple[dict[str, Any], dict[str, Any]]:
  """
  Spools the logs from a lox_p primitive. The logs are extracted from the equation's parameters.

  Args:
      eqn (JaxprEqn): The equation representing the lox_p operation.
  Returns:
      tuple[dict[str, Any], dict[str, Any]]: The logs and their shapes.
  """
  eqn_logs_structure = eqn.params["structure"]
  eqn_logs = jax.tree.unflatten(eqn_logs_structure, eqn.invars)
  eqn_log_shapes_flat = jax.tree.map(lambda v: ShapeDtypeStruct(v.aval.shape, v.aval.dtype), eqn.invars)
  eqn_log_shapes = jax.tree.unflatten(eqn_logs_structure, eqn_log_shapes_flat)
  return eqn_logs, eqn_log_shapes


def spool_scan_p(eqn: JaxprEqn) -> tuple[dict[str, Any], dict[str, Any]]:
  """
  Spools the logs from a scan_p primitive. The logs of the jaxpr are reshaped to have a static length,
  which is the length of the scan.

  Args:
      eqn (JaxprEqn): The equation representing the scan operation.
  Returns:
      tuple[dict[str, Any], dict[str, Any]]: The logs and their shapes.
  """
  jaxpr_logs_shape, jaxpr_log_shapes = spool_jaxpr(eqn.params["jaxpr"].jaxpr)
  eqn_logs = jax.tree.map(lambda s: Var("", aval=ShapedArray((eqn.params["length"],) + s.shape, s.dtype)), jaxpr_logs_shape)
  eqn_log_shapes = jaxpr_log_shapes
  eqn.outvars.extend(jax.tree.leaves(eqn_logs))
  return eqn_logs, eqn_log_shapes


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
  branches_logs, branches_log_shapes = [], []

  for branch in branches:
    branch_logs_shape, branch_log_shapes = spool_jaxpr(branch.jaxpr)
    branches_logs.append(branch_logs_shape)
    branches_log_shapes.append(branch_log_shapes)

  if not all(branches_logs[0] == branch_logs for branch_logs in branches_logs):
    raise ValueError("All branches must have the same log structure.")
  if not all(branches_log_shapes[0] == shape for shape in branches_log_shapes):
    raise ValueError("All branches must have the same log shapes.")

  eqn_logs = jax.tree.map(lambda s: Var("", aval=ShapedArray(s.shape, s.dtype)), branches_logs[0])
  eqn_log_shapes = branches_log_shapes[0]
  eqn.outvars.extend(jax.tree.leaves(eqn_logs))

  return eqn_logs, eqn_log_shapes


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
  jaxpr_logs_shape, jaxpr_log_shapes = spool_jaxpr(eqn.params["jaxpr"].jaxpr)
  if jaxpr_logs_shape or jaxpr_log_shapes:
    raise ValueError("Spooling for while loops is not supported due to non-static length.")
  eqn_logs, eqn_log_shapes = jaxpr_logs_shape, jaxpr_log_shapes
  return eqn_logs, eqn_log_shapes


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
    eqn_logs = jax.tree.map(lambda s: Var("", aval=ShapedArray(s.shape, s.dtype)), jaxpr_logs_shape)
    eqn_log_shapes = jaxpr_log_shapes

    eqn.outvars.extend(jax.tree.leaves(eqn_logs))
    eqn.primitive = jax.extend.core.primitives.call_p
    eqn.params = {"call_jaxpr": eqn.params["jaxpr"].jaxpr}
    return eqn_logs, eqn_log_shapes
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
  jaxpr_logs_shape, jaxpr_log_shapes = spool_jaxpr(eqn.params["call_jaxpr"])
  eqn_logs = jax.tree.map(lambda s: Var("", aval=ShapedArray(s.shape, s.dtype)), jaxpr_logs_shape)
  eqn_log_shapes = jaxpr_log_shapes
  eqn.outvars.extend(jax.tree.leaves(eqn_logs))
  return eqn_logs, eqn_log_shapes
