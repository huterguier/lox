import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jax.tree_util import PyTreeDef 
from typing import Callable


PAD_CHAR = '\uffff'
LENGTH = 1024


@jax.tree_util.register_dataclass
@dataclass
class String:
  def __init__(self, value):
    self.value = value

  def __array__(self):
    return self.value

  def __get_item__(self, key):
    return String(self.value[key])

  def __repr__(self):
    return f'String({self.value})'

  def __str__(self):
    return ''.join(chr(c) for c in self.value if c != ord(PAD_CHAR))

  def __getitem__(self, key):
    return String(self.value[key])

  def __binary_op(self, other, op):
        other_val = other.value if isinstance(other, String) else other
        return String(op(self.value, other_val))

  def __add__(self, other): return self.__binary_op(other, jnp.add)
  def __sub__(self, other): return self.__binary_op(other, jnp.subtract)
  def __mul__(self, other): return self.__binary_op(other, jnp.multiply)


def string(s):
  value = jnp.full((LENGTH,), ord(PAD_CHAR), dtype=jax.numpy.uint8)
  for i, c in enumerate(s):
    if i >= 1024:
      raise ValueError("String too long")
    value = value.at[i].set(ord(c))
  return String(value)


@jax.tree_util.register_dataclass
@dataclass
class ldict(dict):
  pass


def is_hashable(arg):
  """
  Check if an argument is hashable.

  Args:
      arg: The argument to check.
  Returns:
      bool: True if the argument is hashable, False otherwise.
  """
  try:
    hash(arg)
    return True
  except TypeError:
    return False


def flatten(fun: Callable, structure: PyTreeDef) -> Callable:
  """
    Transforms a function to accept a single flat argument list.

    Args:
        fun (Callable): The function to be transformed.
    Returns:
        Callable: A new function that accepts a single flat argument list.
  """
  def wrapped(*args_flat):
    args, kwargs = jax.tree_util.tree_unflatten(structure, args_flat)
    out = fun(*args, **kwargs)
    return out
  return wrapped
