from typing import Any, Iterable

import jax
import jax.numpy as jnp
from jax._src.debugging import DebugEffect
from jax.extend import core
from jax.interpreters import ad, batching, mlir

from lox.logdict import logdict, stepdict

lox_p = core.Primitive("lox")
lox_p.multiple_results = True


def log(
    data: dict[str, Any],
    tags: Iterable[str] = (),
    default: dict[str, Any] | None = None,
    **steps: int,
) -> logdict:
    """
    Fundamental logging primitive for Lox.
    This primitive creates a logdict for a single data point and associates it with the provided steps.

    Args:
        data: A dictionary containing the data to be logged.
        tags: An iterable of strings representing tags associated with the log.
        default: Optional default values used to pad divergent conditional branches.
          Keys must be a subset of ``data``.
        steps: Keyword arguments where keys are step names and values are step numbers.

    Returns:
        logdict: A logdict object containing the logged data and steps.

    Examples:
        By default lox.log complies with the pure functional programming paradigm of JAX,
        meaning it does not have side effects and does not mutate the state.
        Let's look at this simple example of a function that adds 1.0 to its input.

        >>> def f(x):
        >>>     return x + 1.0
        >>> f(1.0)
        2.0

        If we insert a logging statement inside the function, the function's behavior remains unchanged.

        >>> def f_log(x):
        >>>     lox.log({"x": x})
        >>>     return x + 1.0
        >>> f_log(1.0)
        2.0

        In order to actually retrieve or use the logged data, we need to apply a function transformation such as :attr:`spool` or :attr:`tap`.

    """
    if default is None:
        default = {}
    unknown_default_keys = set(default) - set(data)
    if unknown_default_keys:
        raise ValueError(
            f"Default values provided for unknown log keys: {sorted(unknown_default_keys)}"
        )
    for key, value in default.items():
        try:
            hash(value)
        except TypeError as e:
            raise ValueError(
                "Default values must be hashable compile-time constants."
            ) from e
    data_logdict = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), data)
    steps_logdict = {
        key_step: stepdict(
            {key_data: jnp.array([value_step]) for key_data, _ in data.items()}
        )
        for key_step, value_step in steps.items()
    }
    logs = logdict(data_logdict, **steps_logdict)
    logs_flat, structure = jax.tree_util.tree_flatten(logs)
    _, default_structure = jax.tree_util.tree_flatten({})
    default_keys = tuple(data_logdict.keys())
    default_mask = tuple(key in default for key in default_keys)
    default_values = tuple(default.get(key) for key in default_keys)
    _ = lox_p.bind(
        *logs_flat,
        tags=tuple(tags),
        structure=structure,
        default_structure=default_structure,
        default_keys=default_keys,
        default_mask=default_mask,
        default_values=default_values,
        n_logs=len(logs_flat),
        n_defaults=0,
    )
    return jax.tree_util.tree_unflatten(structure, logs_flat)


@lox_p.def_impl
def lox_impl(
    *lox_args_flat,
    tags,
    structure,
    default_structure,
    default_keys,
    default_mask,
    default_values,
    n_logs,
    n_defaults,
):
    del (
        structure,
        tags,
        default_structure,
        default_keys,
        default_mask,
        default_values,
        n_defaults,
    )
    return lox_args_flat[:n_logs]


@lox_p.def_effectful_abstract_eval
def lox_abstract_eval(
    *lox_args_flat,
    tags,
    structure,
    default_structure,
    default_keys,
    default_mask,
    default_values,
    n_logs,
    n_defaults,
):
    del (
        structure,
        tags,
        default_structure,
        default_keys,
        default_mask,
        default_values,
        n_defaults,
    )
    return list(lox_args_flat[:n_logs]), {DebugEffect()}


def lox_lowering(
    *lox_args_flat,
    tags,
    structure,
    default_structure,
    default_keys,
    default_mask,
    default_values,
    n_logs,
    n_defaults,
):
    del (
        structure,
        tags,
        default_structure,
        default_keys,
        default_mask,
        default_values,
        n_defaults,
    )
    return lox_args_flat[:n_logs]


mlir.register_lowering(lox_p, mlir.lower_fun(lox_lowering, multiple_results=True))


def lox_batch(
    vector_arg_values,
    batch_axes,
    tags,
    structure,
    default_structure,
    default_keys,
    default_mask,
    default_values,
    n_logs,
    n_defaults,
):
    outs = lox_p.bind(
        *vector_arg_values,
        tags=tags,
        structure=structure,
        default_structure=default_structure,
        default_keys=default_keys,
        default_mask=default_mask,
        default_values=default_values,
        n_logs=n_logs,
        n_defaults=n_defaults,
    )
    return outs, batch_axes


batching.primitive_batchers[lox_p] = lox_batch


def lox_jvp(
    arg_values,
    arg_tangents,
    tags,
    structure,
    default_structure,
    default_keys,
    default_mask,
    default_values,
    n_logs,
    n_defaults,
):
    lox_p.bind(
        *arg_values,
        tags=tags,
        structure=structure,
        default_structure=default_structure,
        default_keys=default_keys,
        default_mask=default_mask,
        default_values=default_values,
        n_logs=n_logs,
        n_defaults=n_defaults,
    )
    return arg_values, arg_tangents


ad.primitive_jvps[lox_p] = lox_jvp


def lox_p_transpose(ct, x):
    del ct, x
    raise ValueError("Transpose doesn't support logging")


ad.primitive_transposes[lox_p] = lox_p_transpose
