import jax
import jax.core
import jax.extend
from jax import ShapeDtypeStruct
from jax.core import ShapedArray, AxisName
from jax.extend.core import Var, ClosedJaxpr, Jaxpr, JaxprEqn
from typing import Any, Iterable, Sequence, Callable
from lox.primitive import lox_p
from lox.util import is_hashable, flatten
from functools import wraps


def tap(
    fun: Callable, 
    argnames: str | Iterable[str] | None = None,
) -> Callable:
  """
  Spools a function to extract logs and their shapes, allowing for dynamic argument handling.

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
    closed_jaxpr, out_shape = make_tapped_jaxpr(
      flatten(fun, structure),
      static_argnums=static_argnums, 
      return_shape=True,
      argnames=argnames,
    )(*args_flat)
    dynamic_args_flat = tuple(arg for arg in args_flat if not is_hashable(arg))
    out_structure = jax.tree.structure(out_shape)
    out_flat = jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *dynamic_args_flat)
    out = jax.tree_util.tree_unflatten(out_structure, out_flat)
    return out
  return wrapped


def make_tapped_jaxpr(
    fun: Callable,
    static_argnums: int | Iterable[int] = (),
    axis_env: Sequence[tuple[AxisName, int]] | None = None,
    return_shape: bool = False,
    abstracted_axes: Any | None = None,
    argnames: str | Sequence[str] | None = None,
) -> Callable[..., tuple[ClosedJaxpr, Any]]:
  def wrapped(*args, **kwargs):
    closed_jaxpr, out_shape = jax.make_jaxpr(
        fun,
        static_argnums=static_argnums,
        axis_env=axis_env,
        return_shape=True,
        abstracted_axes=abstracted_axes,
    )(*args, **kwargs)
    tap_jaxpr(closed_jaxpr.jaxpr)
  return wrapped


def tap_jaxpr(jaxpr: Jaxpr, argnames: list[str]=None):
  def make_callback(eqn):
    def callback(logs: logdict):
      def log_shape(log):
        if isinstance(log, ShapedArray):
          return log.shape
        elif isinstance(log, ShapeDtypeStruct):
          return log.shape
        else:
          raise TypeError(f"Unsupported log type: {type(log)}")

  i = 0
  while i < len(jaxpr.eqns):
    eqn = jaxpr.eqns[i]
    if eqn.primitive == lox_p:
      # create and equation that handles the printing to the console
      # jaxpr.eqns.insert(i, 
      #   JaxprEqn(
      #     primitive=jax.debug.callback,
      #     inputs=eqn.inputs,
      #     outputs=eqn.outputs,
      #     params=eqn.params,
      #     source_info=source_info_util.current(),
      #   )
      # )
    elif eqn.primitive == jax.lax.scan_p:
      tap_jaxpr(eqn.params["jaxpr"].jaxpr, argnames)
    elif eqn.primitive == jax.lax.cond_p:
      branches = eqn.params["branches"]
      for branch in branches:
        tap_jaxpr(branch.jaxpr, argnames)
    elif eqn.primitive == jax.lax.while_p:
      tap_jaxpr(eqn.params["cond_jaxpr"].jaxpr, argnames)
      tap_jaxpr(eqn.params["body_jaxpr"].jaxpr, argnames)
    elif eqn.primitive.name == "pjit":
      tap_jaxpr(eqn.params["jaxpr"].jaxpr)
    elif eqn.primitive == jax.extend.core.primitives.call_p:
      tap_jaxpr(eqn.params["call_jaxpr"].jaxpr)
