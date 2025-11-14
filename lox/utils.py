from typing import Any, Callable

import jax

from lox.typing import Key


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
        structure (Any): The structure used to flatten and unflatten the arguments.
    Returns:
        Callable: A new function that accepts a single flat argument list.
    """

    def wrapped(*args_flat):
        args, kwargs = jax.tree_util.tree_unflatten(structure, args_flat)
        out = fun(*args, **kwargs)
        return out

    return wrapped


def get_path(path: str, key: Key) -> str:
    """
    Constructs a new path by appending a key to an existing path.

    Args:
        path (str): The base path.
        key (Key): The key to append to the path.
    Returns:
        str: The new constructed path.
    """
    key_data = jax.random.key_data(key)
    folder_name = str(int(f"{key_data[0]}{key_data[1]}"))
    path = path + "/" + folder_name
    return path
