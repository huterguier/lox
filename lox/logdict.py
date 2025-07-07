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
  This class extends the standard dictionary to include step information,
  allowing for structured logging of data during computations.
  It is the underlying data structure used for all logging in lox.
  It behaves identically to a standard dictionary, but it additionally contains
  a ``steps`` attribute that stores timestamps at which the data was logged.
  Internally ``steps`` two level dictionary, where at the first level are
  the names of the timestamps (e.g. "step", "episode", etc.),
  and at the second level are the actual timestamps.
  However, it is not supposed to be accessed directly.
  Insted ``logdict`` provides a convenient interface to access the steps
  as attributes.


  .. code-block:: python

    >>> _, logs = lox.spool(f)()
    >>> logs["loss"]
    [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.2]
    >>> logs.step["loss"]
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    >>> loss.episode["loss"]
    [0, 0, 0, 1, 1, 1, 2, 2, 2]

  """
  steps: dict[str, stepdict]


  def __init__(self, data: dict[str, Any], **steps: dict[str, jax.Array]):
    super().__init__(data)
    self.steps = {k: stepdict(v) for k, v in steps.items()}


  def tree_flatten(self) -> tuple:
    """
    Function that flattens the logdict into a flat list of data and steps.
    This is used to register the logdict as a pytree in JAX.

    Returns:
      tuple: A tuple containing the flattened data and steps.
      The first element is a flat list of data values, and the second element is a tuple
      containing the structure of the data and steps.
    """
    data_flat, data_structure = jax.tree_util.tree_flatten(self.data)
    steps_flat, steps_structure = jax.tree_util.tree_flatten(self.steps)
    return data_flat + steps_flat, (data_structure, steps_structure)


  def tree_unflatten(self, aux_data: tuple) -> 'logdict':
    """
    Function that reconstructs the logdict from a flat list of data and steps.
    This is used to register the logdict as a pytree in JAX.

    Args:
      aux_data (tuple): A tuple containing the flattened data and steps.
        The first element is a flat list of data values, and the second element is a tuple
        containing the structure of the data and steps.
    Returns:
      logdict: A new logdict instance containing the reconstructed data and steps.

    """
    data_flat, steps_flat = aux_data
    data = jax.tree_util.tree_unflatten(data_flat, self.data)
    steps = jax.tree_util.tree_unflatten(steps_flat, self.steps)
    return logdict(data, **steps)


  def __getattr__(self, item):
    if item in self.steps:
      return self.steps[item]
    raise AttributeError(f"'logdict' object has no attribute '{item}'")


  @property
  def data(self):
    """
    Returns:
      dict: The data stored in the logdict as a standard dictionary.
    """
    return dict(self)


  def __or__(self, other):
    """
    Merges two logdicts, overwriting values from the self dict if they exist in both.
    The same happens for the steps.

    Args:
      other (logdict): Another logdict to merge with.
    Returns:
      logdict: A new logdict containing the merged data and steps.
    Raises:
      TypeError: If the other object is not a logdict.
    """
    if not isinstance(other, logdict):
      raise TypeError("can only merge with another logdict")
    out = super().__or__(other)
    out.steps = self.steps | other.steps


  def __add__(self, other):
    """
    Adds two logdicts together, concatenating their data and steps.
    This is essentially identical to executing two functions in sequence and then collecting their logs.
    Assuming ``f`` and ``g`` are two pure functions containing arbitrary logs, then contents
    of the variable ``logs`` will be the same in the following two codeblocks.

    .. code-block:: python

        _, logs_f = lox.spool(f)()
        _, logs_g = lox.spool(g)()

        logs = logs_f + logs_g

    .. code-block:: python

        def h():
            f()
            g()
        _, logs = lox.spool(h)()

    Note that this is not the same as updating the dict.
    If both logdicts contain values for the same key, the values will be concatenated,
    assuming they have the same structure. If they do not, an error will be raised.

    Args:
      other (logdict): Another logdict to add.
    Returns:
      logdict: A new logdict containing the concatenated data and steps.
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


