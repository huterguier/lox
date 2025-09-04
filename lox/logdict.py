import jax
import jax.numpy as jnp
from typing import Any, Callable, Optional


@jax.tree_util.register_pytree_node_class
class stepdict(dict[str, jax.Array]):
    """
    A dictionary that stores step values.
    """

    def __init__(self, data: dict, **kwargs):
        super().__init__(data)
        for key, value in kwargs.items():
            self[key] = jnp.asarray(value)

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

    def __add__(self, other: "stepdict") -> "stepdict":
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

    def reduce(self, mode: str = "mean") -> "stepdict":
        """
        Reduces the stepdict values using the specified mode.
        Args:
          mode (str): The reduction mode to apply. Can be one of "mean", "first" or "last".
        Returns:
          stepdict: A new stepdict containing the reduced values.
        """
        if mode == "mean":
            return stepdict(
                {
                    k: jnp.mean(v, keepdims=True) if len(v) > 1 else v
                    for k, v in self.items()
                }
            )
        if mode == "first":
            return stepdict({k: v[:1] for k, v in self.items()})
        if mode == "last":
            return stepdict({k: v[-1:] for k, v in self.items()})
        else:
            raise ValueError(f"Unknown reduction mode: {mode}")

    def _slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> "stepdict":
        """
        Slices the stepdict values.
        Args:
          start (int | None): The start index of the slice.
          stop (int | None): The stop index of the slice.
          step (int | None): The step size of the slice.
        Returns:
          stepdict: A new stepdict containing the sliced values.
        """
        return stepdict({k: v[start:stop:step] for k, v in self.items()})

    class _SliceProxy:
        def __init__(self, stepdict: "stepdict"):
            self.stepdict = stepdict

        def __getitem__(self, key: slice) -> "stepdict":
            if not isinstance(key, slice):
                raise TypeError("SliceProxy only supports slicing with slice objects.")
            return self.stepdict._slice(key.start, key.stop, key.step)

    @property
    def slice(self) -> "_SliceProxy":
        """
        Provides a convenient interface to slice the stepdict.
        This allows to slice the stepdict using the standard slicing syntax.

        Returns:
          _SliceProxy: A proxy object that allows slicing the stepdict.
        """
        return self._SliceProxy(self)

    def tree_flatten(self) -> tuple:
        """
        Flattens the stepdict into a flat list of values.
        This is used to register the stepdict as a pytree in JAX.

        Returns:
          tuple: A tuple containing the flattened values and the structure of the stepdict.
        """
        return jax.tree_util.tree_flatten(dict(self))

    @classmethod
    def tree_unflatten(cls, structure, steps_flat) -> "stepdict":
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

    def __init__(self, data: dict[str, Any], **steps: stepdict):
        super().__init__(data)
        self.steps = steps

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
    def tree_unflatten(cls, structure: tuple, logs_flat) -> "logdict":
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
        data_flat = logs_flat[: data_structure.num_leaves]
        steps_flat = logs_flat[data_structure.num_leaves :]
        data = jax.tree_util.tree_unflatten(data_structure, data_flat)
        steps = jax.tree_util.tree_unflatten(steps_structure, steps_flat)
        return cls(data, **steps)

    def __delitem__(self, key):
        """
        Deletes an item from the logdict.
        This behaves like a standard dictionary, but also removes the corresponding step information.

        Args:
          key (str): The key of the item to delete.
        Raises:
          KeyError: If the key does not exist in the logdict.
        """
        super().__delitem__(key)
        for step in self.steps.values():
            if key in step:
                del step[key]

    def __getattr__(self, item):
        if item in self.steps:
            return self.steps[item]
        raise AttributeError(f"'logdict' object has no attribute '{item}'")

    @property
    def data(self) -> dict[str, Any]:
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
        new_data = super().__or__(other)
        new_steps = {}
        for key in set(self.steps.keys()).union(other.steps.keys()):
            if key in self.steps and key in other.steps:
                new_steps[key] = self.steps[key] | other.steps[key]
            elif key in self.steps:
                new_steps[key] = self.steps[key]
            elif key in other.steps:
                new_steps[key] = other.steps[key]
        return logdict(new_data, **new_steps)

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
        for key in set(self.keys()).union(other.keys()):
            if key in self and key in other:
                new_data[key] = jnp.concatenate((self[key], other[key]))
            elif key in self:
                new_data[key] = self[key]
            elif key in other:
                new_data[key] = other[key]
        new_steps = {}
        for key in set(self.steps.keys()).union(other.steps.keys()):
            if key in self.steps and key in other.steps:
                new_steps[key] = self.steps[key] + other.steps[key]
            elif key in self.steps:
                new_steps[key] = self.steps[key]
            elif key in other.steps:
                new_steps[key] = other.steps[key]

        return logdict(new_data, **new_steps)

    def reduce(
        self,
        mode: str = "mean",
        keep_steps: bool = True,
    ) -> "logdict":
        """
        Reduces the logdict values using the specified mode.

        Args:
            mode (str): The reduction mode to apply. Can be one of "mean", "first" or "last".
            keep_steps (bool): Whether to keep the steps in the reduced logdict.
        Returns:
            logdict: A new logdict containing the reduced values and optionally the steps.
        Note that reduction over specific steps is not implemented yet.

        Raises:
            ValueError: If the reduction mode is not recognized.
        """
        if mode == "mean":
            data_reduced = {
                k: jnp.mean(v, keepdims=True) if len(v) > 1 else v
                for k, v in self.items()
            }
            steps_reduced = {k: v.reduce(mode) for k, v in self.steps.items()}
            return logdict(data_reduced, **steps_reduced if keep_steps else {})
        if mode == "first":
            data_reduced = {k: v[:1] for k, v in self.items()}
            steps_reduced = {k: v.reduce(mode) for k, v in self.steps.items()}
            return logdict(data_reduced, **steps_reduced if keep_steps else {})
        if mode == "last":
            data_reduced = {k: v[-1:] for k, v in self.items()}
            steps_reduced = {k: v.reduce(mode) for k, v in self.steps.items()}
            return logdict(data_reduced, **steps_reduced if keep_steps else {})
        else:
            raise ValueError(f"Unknown reduction mode: {mode}")

    def _slice(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ):
        """
        Slices the logdict values and steps.

        Args:
          start (int | None): The start index of the slice.
          stop (int | None): The stop index of the slice.
          step (int | None): The step size of the slice.
        Returns:
            logdict: A new logdict containing the sliced values and steps.
        """
        return logdict(
            {k: jax.tree.map(lambda x: x[start:stop:step], v) for k, v in self.items()},
            **{k: v.slice[start:stop:step] for k, v in self.steps.items()},
        )

    class _SliceProxy:
        def __init__(self, logdict: "logdict"):
            self.logdict = logdict

        def __getitem__(self, key: slice) -> "logdict":
            if not isinstance(key, slice):
                raise TypeError("SliceProxy only supports slicing with slice objects.")
            return self.logdict._slice(key.start, key.stop, key.step)

    @property
    def slice(self) -> "_SliceProxy":
        """
        Provides a convenient interface to slice the logdict.
        This allows to slice the logdict using the standard slicing syntax.

        Returns:
          _SliceProxy: A proxy object that allows slicing the logdict.
        """
        return self._SliceProxy(self)
