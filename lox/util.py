import jax
import jax.numpy as jnp
from dataclasses import dataclass


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
