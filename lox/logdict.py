import jax
import jax.numpy as jnp
from typing import Any, Dict


class stepdict(dict[str, jax.Array]):
  """
  A dictionary that stores step values.
  """

  def __init__(self, data: dict, **kwargs):
    super().__init__(data)
    self.steps = {}
    for key, value in kwargs.items():
      self.steps[key] = jnp.asarray(value)

  def __getattr__(self, item):
    if item in self:
      return self[item]
    raise AttributeError(f"'stepdict' object has no attribute '{item}'")

  def __or__(self, other):
    """
    Merges two stepdicts, overwriting values from the self dict if they exist in both.
    """
    if not isinstance(other, stepdict):
      raise TypeError("Can only merge with another stepdict")
    out = super().__or__(other)
    return out

  def __add__(self, other: 'stepdict') -> 'stepdict':
    """
    Adds two stepdicts together, concatenating their values.
    """
    if not isinstance(other, stepdict):
      raise TypeError("Can only add another stepdict")
    new_steps = {}
    for key in set(self.keys()).union(other.keys()):
      if key in self and key in other:
        new_steps[key] = jnp.concatenate((self[key], other[key]))
      elif key in self:
        new_steps[key] = self[key]
      elif key in other:
        new_steps[key] = other[key]
    return stepdict(new_steps)


@jax.tree_util.register_pytree_node_class
class logdict(dict[str, Any]):
  """
  A dictionary that stores log values and the steps at which they were recorded.
  """
  steps: dict[str, stepdict]


  def __init__(self, data: dict[str, any], **steps: dict[str, jax.Array]):
    super().__init__(data)
    self.steps = {k: stepdict(v) for k, v in steps.items()}


  def tree_flatten(self):
    """
    flatten the logdict into a tuple of (data, steps).
    """
    data_flat, data_structure = jax.tree_util.tree_flatten(self.data)
    steps_flat, steps_structure = jax.tree_util.tree_flatten(self.steps)
    return data_flat + steps_flat, (data_structure, steps_structure)


  def tree_unflatten(self, aux_data):
    data_flat, steps_flat = aux_data
    data = jax.tree_util.tree_unflatten(data_flat, self.data)
    steps = jax.tree_util.tree_unflatten(steps_flat, self.steps)
    return logdict(data, **steps)


  def __getattr__(self, item):
    print(f"accessing attribute '{item}' in logdict")
    if item in self.steps:
      return self.steps[item]
    raise attributeerror(f"'logdict' object has no attribute '{item}'")


  @property
  def data(self):
    return dict(self)


  def __or__(self, other):
    """
    merges two logdicts, overwriting values from the self dict if they exist in both.
    """
    if not isinstance(other, logdict):
      raise typeerror("can only merge with another logdict")
    out = super().__or__(other)
    out.steps = self.steps | other.steps


  def __add__(self, other):
    """
    Adds two logdicts together, concatenating their data and steps.
    This is essentially identical to executing two functions in sequence and then collecting their logs.
    Assuming `f` and `g` are two pure functions containing arbitrary logs, then the following holds.

      >>> _, logs_f = lox.spool(f)(...)
      >>> _, logs_g = lox.spool(g)(...)
      >>> logs = logs_f + logs_g

      >>> def h(...):
      >>>   f(...)
      >>>   g(...)
      >>> _, logs = lox.spool(h)

    """
    if not isinstance(other, logdict):
      raise TypeError("Can only add another logdict")
    # iterate the keys and values side by side
    new_data = {}
    new_steps = self.steps + other.steps
    for key in set(self.keys()).union(other.keys()):
      if key in self and key in other:
        new_data[key] = jnp.concatenate((self[key], other[key]))
      elif key in self:
        new_data[key] = self[key]
      elif key in other:
        new_data[key] = other[key]
    new_steps = self.steps + other.steps

    return logdict(new_data, **new_steps)


data = {
    "loss": jnp.array([0.1, 0.2, 0.3]),
    "x": jnp.array([1.0, 2.0, 3.0]),
}
logs = logdict(data, step={"loss": jnp.array([0, 1, 2]), "x": jnp.array([0, 1, 2])})
print(logs.data)
print(logs.step["x"])

logs_flat, structure = jax.tree_util.tree_flatten(logs)
print(structure)


