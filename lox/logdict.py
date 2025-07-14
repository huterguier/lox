import jax
import jax.numpy as jnp
from typing import Any, Callable


@jax.tree_util.register_pytree_node_class
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

  def tree_flatten(self) -> tuple:
    """
    Flattens the stepdict into a flat list of values.
    This is used to register the stepdict as a pytree in JAX.

    Returns:
      tuple: A tuple containing the flattened values and the structure of the stepdict.
    """
    return jax.tree_util.tree_flatten(self)


  @classmethod
  def tree_unflatten(cls, structure, steps_flat) -> 'stepdict':
    """
    Reconstructs the stepdict from a flat list of values.
    This is used to register the stepdict as a pytree in JAX.

    Args:
      structure (tuple): The structure of the stepdict.
      flat_values: A flat list of values to reconstruct the stepdict.
    Returns:
      stepdict: A new stepdict instance containing the reconstructed values.
    """
    return cls(jax.tree_util.tree_unflatten(structure, steps_flat))


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
  the names of the timestamps (e.g. :string:`step`, :string:`episode`, etc.),
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
    for k, v in self.steps.items():
      print(type(v))


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


  @classmethod
  def tree_unflatten(cls, structure: tuple, logs_flat) -> 'logdict':
    """
    Function that reconstructs the logdict from a flat list of data and steps.
    This is used to register the logdict as a pytree in JAX.

    Args:
      structure (tuple): A tuple containing the structure of the data and steps.
      logs_flat: A flat list of data values and steps.
    Returns:
      logdict: A new logdict instance containing the reconstructed data and steps.
    """
    data_structure, steps_structure = structure
    data_flat = logs_flat[:data_structure.num_leaves]
    steps_flat = logs_flat[data_structure.num_leaves:]
    data = jax.tree_util.tree_unflatten(data_structure, data_flat)
    steps = jax.tree_util.tree_unflatten(steps_structure, steps_flat)
    return cls(data, **steps)


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
          f(); g()
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


  def reduce(
      self, 
      mode: str | Callable = "mean", 
      step: str | None = None,
      keep_steps: bool = True
  ) -> 'logdict':
    """
    Reduces the logdict data using the specified mode.
    The ``step`` argument allows to specify on which step the reduction should be applied.
    Whenever there are other steps, the reduction will be applied to them aswell.
    Alternatively you can set ``keep_steps`` to ``False`` to remove all other steps.



    Args:
      mode (str | Callable): The reduction mode to apply. Can be one of "mean", "sum", or a custom function.
      step (str): The step on which to apply the reduction. If None, the reduction is applied to all steps.
      keep_steps (bool): Whether to keep the all other steps in the logdict.
    Returns:
      logdict: A new logdict containing the reduced data.

    Examples:
      The following example shows how to reduce the loss values by taking the mean over
      the :string:`episode` step, while keeping the other steps intact.
      First we create a logdict with some dummy data.

      >>> _, logs = lox.spool(f)()
      >>> logs["loss"]
      [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.17]
      >>> logs.step["loss"]
      [0, 1, 2, 3, 4, 5, 6, 7, 8]
      >>> loss.episode["loss"]
      [0, 0, 0, 1, 1, 1, 2, 2, 2]

      Now we can reduce the loss values by taking the mean over the "mean" step.

      >>> logs = logs.reduce("mean", step="episode", keep_steps=True)
      >>> logs["loss"]
      [0.8, 0.4, 0.21]
      >>> logs.step["loss"]
      [1, 4, 7]
      >>> logs.episode["loss"]
      [0, 1, 2]

    This example shows how to reduce the loss values by taking the mean
    over the :string:`episode` step, while keeping the other steps intact.
    Consequently the reduction is also applied to the :string:`step` step, where 
    the mean is taken over all steps that belong to the same episode.
    There are also other reduction modes available, such as :string:`max` or a "min".
    But note that these reduction modes can only be applied to scalar values,
    as they do not make sense for general arrays or pytrees.
    """



    return self
