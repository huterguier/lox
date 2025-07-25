import jax
import jax.core
import jax.extend
from jax.extend.core import ClosedJaxpr, Jaxpr
from typing import Any, Hashable, Iterable, Sequence, Callable
from lox.primitive import lox_p
from lox.util import is_hashable, flatten
from functools import wraps

AxisName = Hashable

def tap(
    fun: Callable, 
    argnames: str | Iterable[str] | None = None,
) -> Callable:
  """
  A decorator that taps into the execution of a JAX function and prints the values of specified arguments. One can only `tap` into values that are logged with `lox.log`.

  Args:
    fun: The JAX function to be tapped.
    argnames: A string or iterable of strings specifying the names of the arguments to be printed. If None, all arguments will be tapped.
  """
  @wraps(fun)
  def wrapped(*args, **kwargs):
    args_flat, structure = jax.tree_util.tree_flatten((args, kwargs))
    static_argnums = tuple(i for i, arg in enumerate(args_flat) if is_hashable(arg))
    closed_jaxpr, out_shape = make_tapped_jaxpr(
      flatten(fun, structure),
      static_argnums=static_argnums, 
      return_shape=True,
      argnames=argnames,
    )(*args_flat)
    dynamic_args_flat = tuple(arg for arg in args_flat if not is_hashable(arg))
    out_structure = jax.tree_util.tree_structure(out_shape)
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
    modified = tap_jaxpr(closed_jaxpr.jaxpr, argnames=argnames)
    del modified  # Unused, might throw a warning if there are no modifications in the future.
    return closed_jaxpr, out_shape
  return wrapped


def tap_jaxpr(jaxpr: Jaxpr, argnames: Iterable[str] | None = None):
  """
  Taps into a JAX Jaxpr and prints the values of specified arguments. Recurisvely traverses the Jaxpr to find and tap into `lox_p` primitives.

  Args:
    jaxpr (Jaxpr): The Jaxpr to be tapped.
    argnames (Iterable[str] | None): An iterable of argument names to be tapped. If None, all arguments will be tapped.
  Returns:
    bool: True if the Jaxpr was modified, False otherwise.
  """
  def callback(structure, *vals):
    def _callback(*vals):
      data = jax.tree_util.tree_unflatten(structure, vals)
      print(data)
    jax.debug.callback(
        _callback,
        *vals,
    )

  i = 0
  modified = False

  while i < len(jaxpr.eqns):
    eqn = jaxpr.eqns[i]

    if eqn.primitive == lox_p:
      structure = eqn.params["structure"]
      invars_data = jax.tree_util.tree_unflatten(structure, eqn.invars)
      if argnames is None:
        invars_data_tapped = invars_data
      else:
        invars_data_tapped = {k: v for k, v in invars_data.items() if k in argnames}
      invars_tapped, structure_tapped = jax.tree_util.tree_flatten(invars_data_tapped)
      avals_tapped = jax.tree_util.tree_map(lambda var: var.aval, invars_tapped)
      if avals_tapped:
        print_jaxpr = jax.make_jaxpr(callback, static_argnums=(0,))(structure_tapped, *avals_tapped)
        jaxpr.eqns.insert(i, print_jaxpr.jaxpr.eqns[0])
        jaxpr.eqns[i].invars = invars_tapped
        i += 1
        modified = True

    elif eqn.primitive == jax.lax.scan_p:
      modified |= tap_jaxpr(eqn.params["jaxpr"].jaxpr, argnames)

    elif eqn.primitive == jax.lax.cond_p:
      branches = eqn.params["branches"]
      for branch in branches:
        modified |= tap_jaxpr(branch.jaxpr, argnames)

    elif eqn.primitive == jax.lax.while_p:
      modified |= tap_jaxpr(eqn.params["cond_jaxpr"].jaxpr, argnames)
      modified |= tap_jaxpr(eqn.params["body_jaxpr"].jaxpr, argnames)

    elif eqn.primitive.name == "pjit":
      call_jaxpr_modified = tap_jaxpr(eqn.params["jaxpr"], argnames)
      if call_jaxpr_modified:
        eqn.primitive = jax.extend.core.primitives.call_p
        print(eqn.params)
        eqn.params = {"call_jaxpr": eqn.params["jaxpr"].jaxpr}
        modified = True

    elif eqn.primitive == jax.extend.core.primitives.call_p:
      modified |= tap_jaxpr(eqn.params["call_jaxpr"])

    i += 1

    return modified
