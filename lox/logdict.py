from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class logdict(dict[str, Any]):
    """
    A dictionary that stores log values.
    This class extends the standard dictionary and is registered as a JAX pytree,
    allowing it to flow through JAX transformations.
    It supports concatenation (``+``), merging (``|``), slicing (``.slice``),
    reduction (``.reduce``), filtering (``.filter``), and prefixing (``.prefix``).

    .. code-block:: python

      >>> _, logs = lox.spool(f)()
      >>> logs["loss"]
      Array([1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.2], dtype=float32)

    """

    def tree_flatten(self) -> tuple:
        return jax.tree_util.tree_flatten(dict(self))

    @classmethod
    def tree_unflatten(cls, structure, logs_flat) -> "logdict":
        return cls(jax.tree_util.tree_unflatten(structure, logs_flat))

    def __or__(self, other):
        """Merges two logdicts, overwriting values from self with values from other."""
        if not isinstance(other, logdict):
            raise TypeError("can only merge with another logdict")
        return logdict(super().__or__(other))

    def __add__(self, other):
        """
        Concatenates two logdicts along the leading axis.
        Keys present in only one operand are kept as-is; shared keys are concatenated.
        """
        if not isinstance(other, logdict):
            raise TypeError("Can only add another logdict")
        new_data = {}
        for key in set(self.keys()).union(other.keys()):
            if key in self and key in other:
                new_data[key] = jnp.concatenate((self[key], other[key]))
            elif key in self:
                new_data[key] = self[key]
            else:
                new_data[key] = other[key]
        return logdict(new_data)

    def reduce(self, mode: str = "mean") -> "logdict":
        """
        Reduces all values along the leading axis.

        Args:
            mode: One of ``"mean"``, ``"first"``, or ``"last"``.
        """
        if mode == "mean":
            return logdict(
                {k: jnp.mean(v, keepdims=True) if len(v) > 1 else v for k, v in self.items()}
            )
        if mode == "first":
            return logdict({k: v[:1] for k, v in self.items()})
        if mode == "last":
            return logdict({k: v[-1:] for k, v in self.items()})
        raise ValueError(f"Unknown reduction mode: {mode}")

    def _slice(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> "logdict":
        return logdict(
            {k: jax.tree.map(lambda x: x[start:stop:step], v) for k, v in self.items()}
        )

    class _SliceProxy:
        def __init__(self, logdict: "logdict"):
            self.logdict = logdict

        def __getitem__(self, key: slice) -> "logdict":
            return self.logdict._slice(key.start, key.stop, key.step)

    @property
    def slice(self) -> "_SliceProxy":
        """Slice all values: ``logs.slice[::10]``."""
        return self._SliceProxy(self)

    def filter(self, predicate: Callable[[str, Any], bool]) -> "logdict":
        """Returns a new logdict containing only items where ``predicate(key, value)`` is True."""
        return logdict({k: v for k, v in self.items() if predicate(k, v)})

    def prefix(self, prefix: str) -> "logdict":
        """Returns a new logdict with ``prefix`` prepended to every key."""
        return logdict({f"{prefix}{k}": v for k, v in self.items()})
