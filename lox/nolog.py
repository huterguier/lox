import functools
from typing import Callable, Hashable

import jax
from jax._src import source_info_util
from jax.core import ShapedArray
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Var

from lox.primitive import lox_p

AxisName = Hashable

AxisName = Hashable


def strip(fun: Callable) -> Callable:
  """
  Strips all logging operations from the given function by manipulating its Jaxpr.

  Args:
    fun: The function from which to strip logging operations.
  Returns: 
    A new function with logging operations removed.
  """
  @functools.wraps(fun)
  def wrapped(*args, **kwargs):
    closed_jaxpr, out_shape = jax.make_jaxpr(fun, return_shape=True)(*args, **kwargs)
    strip_jaxpr(closed_jaxpr.jaxpr)
    out_flat = jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
    out = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(out_shape), out_flat)
    return out
  return wrapped


def strip_jaxpr(jaxpr: Jaxpr) -> None:
  """Remove all logging operations from a Jaxpr."""
  eqns_log = []
  for eqn in jaxpr.eqns:
    if eqn.primitive == lox_p:
      eqns_log.append(eqn)
    elif eqn.primitive == jax.extend.core.primitives.scan_p:
      strip_jaxpr(eqn.params['jaxpr'])
    elif eqn.primitive == jax.extend.core.primitives.cond_p:
      for branch in eqn.params['branches']:
        strip_jaxpr(branch.jaxpr)
    elif eqn.primitive == jax.extend.core.primitives.while_p:
      strip_jaxpr(eqn.params['body_jaxpr'])
      strip_jaxpr(eqn.params['cond_jaxpr'])
    elif eqn.primitive == jax.extend.core.primitives.jit_p:
      strip_jaxpr(eqn.params['jaxpr'])
    elif eqn.primitive == jax.extend.core.primitives.call_p:
      strip_jaxpr(eqn.params['call_jaxpr'])
    elif eqn.primitive == jax._src.ad_checkpoint.remat_p:
      strip_jaxpr(eqn.params['jaxpr'])
  
  for eqn in eqns_log:
    jaxpr.eqns.remove(eqn)
  

