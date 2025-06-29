import jax
import jax.numpy as jnp


class logdict(dict):
  """
  A dictionary that stores log values and the steps at which they were recorded.
  """
  steps: stepdict


  def __init__(self, data: dict, **kwargs):
    super().__init__(data)
    self.steps = {}
    for key, value in kwargs.items():
      steps = jnp.asarray(value)


  def __getattr__(self, item):
    if item in self:
      return self[item]
    raise AttributeError(f"'logdict' object has no attribute '{item}'")


  @property
  def data(self):
    return dict(self)


  def __or__(self, other):
    """
    Merges two logdicts, overwriting values from the self dict if they exist in both.
    """
    if not isinstance(other, logdict):
      raise TypeError("Can only merge with another logdict")
    out = super().__or__(other)
    out.steps = self.steps | other.steps


  def __add__(self, other):
    """
    Adds two logdicts together, concatenating their data and steps.
    """
    if not isinstance(other, logdict):
      raise TypeError("Can only add another logdict")
    # iterate the keys and values side by side
    new_data = {}
    for key in set(self.keys()).union(other.keys()):
      if key in self and key in other:
        new_data[key] = jnp.concatenate((self[key], other[key]))
        new_steps[key] = jnp.concatenate((self.steps.get(key, []), other.steps.get(key, [])))
      elif key in self:
        new_data[key] = self[key]
        new_steps[key] = self.steps.get(key, [])
      elif key in other:
        new_data[key] = other[key]
        new_steps[key] = other.steps.get(key, [])
    new_steps = self.steps + other.steps

    return logdict(new_data, **new_steps)


class stepdict(dict):
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
