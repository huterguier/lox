from typing import Any, Callable

import jax


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


def flatten(fun: Callable, structure: Any) -> Callable:
    """
    Transforms a function to accept a single flat argument list,
    where the arguments are flattened according to the provided structure.

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
