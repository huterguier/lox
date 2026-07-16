from typing import Any, Callable, Iterable, Optional

import jax
from jax import Array as Key

from lox.logdict import logdict

# from lox.utils.typing import Key


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


def select_logs(
    logs: "logdict",
    eqn_tags: Iterable[str],
    argnames: Optional[str | Iterable[str]],
    tags: Optional[str | Iterable[str]],
) -> "logdict":
    """
    Selects the subset of ``logs`` matching ``argnames``/``tags``.

    ``None`` means no restriction on that axis; a list (even an empty one) restricts
    to exactly what's given, so ``argnames=[]`` or ``tags=[]`` select nothing. When
    both ``argnames`` and ``tags`` are given, a log entry is selected only if it
    matches both (AND). A bare string is treated as a single name/tag, not an
    iterable of characters (mirroring ``jax.jit``'s handling of ``static_argnames``).

    Args:
        logs (logdict): The logs produced by a single ``lox.log`` call.
        eqn_tags (Iterable[str]): The tags associated with that call.
        argnames (Optional[str | Iterable[str]]): Key(s) to restrict the selection to.
        tags (Optional[str | Iterable[str]]): Tag(s) to restrict the selection to.
    Returns:
        logdict: The selected subset of ``logs``.
    """
    if isinstance(argnames, str):
        argnames = (argnames,)
    if isinstance(tags, str):
        tags = (tags,)
    if tags is not None and not any(tag in tags for tag in eqn_tags):
        return logdict({})
    if argnames is not None:
        return logs.filter(lambda k, _: k in argnames)
    return logs


def get_path(path: str, key: Key) -> str:
    """
    Constructs a new path by appending a key to an existing path.

    The folder name is derived by packing the key's raw words into a single integer
    (each word shifted into its own 32-bit slot), which is a bijection -- distinct keys
    always produce distinct folder names. For simple keys made with ``jax.random.key(n)``
    (whose raw data is ``[0, n]``), this reduces to exactly ``str(n)``.

    Args:
        path (str): The base path.
        key (Key): The key to append to the path.
    Returns:
        str: The new constructed path.
    """
    key_data = jax.random.key_data(key)
    combined = 0
    for x in key_data.flatten():
        combined = (combined << 32) | int(x)
    path = path + "/" + str(combined)
    return path
