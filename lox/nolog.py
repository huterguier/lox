import jax
from typing import Callable
from jax.core import ShapedArray, AxisName
from jax.extend.core import Var, ClosedJaxpr, Jaxpr, JaxprEqn
from jax._src import source_info_util
from lox.primitive import lox_p
import functools


def nolog(fun: Callable) -> Callable:
  @functools.wraps(fun)
  def wrapped(*args, **kwargs):
    closed_jaxpr, out_shape = jax.make_jaxpr(fun, return_shape=True)(*args, **kwargs)
    nolog_jaxpr(closed_jaxpr.jaxpr)
    out_structure = jax.tree_util.tree_structure(out_shape)
    out_flat = jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
    out = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(out_shape), out_flat)
    return out
  return wrapped


def nolog_jaxpr(jaxpr: Jaxpr) -> None:
  """Remove all logging operations from a Jaxpr."""
  log_eqns = []
  for eqn in jaxpr.eqns:
    if eqn.primitive == lox_p:
      log_eqns.append(eqn)
    if eqn.primitive == jax.lax.scan_p:
      nolog_jaxpr(eqn.params['jaxpr'])
    elif eqn.primitive == jax.lax.cond_p:
      for branch in eqn.params['branches']:
        nolog_jaxpr(branch.jaxpr)
    elif eqn.primitive == jax.lax.while_p:
      nolog_jaxpr(eqn.params['body_jaxpr'])
      nolog_jaxpr(eqn.params['cond_jaxpr'])
    elif eqn.primitive.name == "pjit":
      nolog_jaxpr(eqn.params['jaxpr'])
    elif eqn.primitive == jax.extend.core.primitives.call_p:
      nolog_jaxpr(eqn.params['call_jaxpr'])
  
  for eqn in log_eqns:
    jaxpr.eqns.remove(eqn)
  

