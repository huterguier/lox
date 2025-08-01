import jax
import jax.numpy as jnp
from jax.extend import core
from jax.interpreters import mlir, batching, ad
from jax._src.debugging import DebugEffect
from lox.logdict import logdict, stepdict


lox_p = core.Primitive("lox")
lox_p.multiple_results = True


def log(data: dict, **steps: int) -> logdict:
    data_logdict = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), data)
    steps_logdict = {
        key_step: stepdict({key_data: value_step for key_data, _ in data.items()})
        for key_step, value_step in steps.items()
    }
    logs = logdict(data_logdict, **steps_logdict)
    logs_flat, structure = jax.tree_util.tree_flatten(logs)
    _ = lox_p.bind(*logs_flat, structure=structure)
    return jax.tree_util.tree_unflatten(structure, logs_flat)

@lox_p.def_impl
def lox_impl(*logs_flat, structure):
    del structure
    return logs_flat

@lox_p.def_effectful_abstract_eval
def lox_abstract_eval(*logs_flat, structure):
    del structure
    return list(logs_flat), {DebugEffect()}

def lox_lowering(*logs_flat, structure):
    del structure
    return logs_flat
mlir.register_lowering(lox_p, mlir.lower_fun(lox_lowering, multiple_results=True))

def lox_batch(vector_arg_values, batch_axes, structure):
    outs = lox_p.bind(*vector_arg_values, structure=structure)
    return outs, batch_axes
batching.primitive_batchers[lox_p] = lox_batch

def lox_jvp(arg_values, arg_tangents, structure):
    lox_p.bind(*arg_values, structure=structure)
    return arg_values, arg_tangents
ad.primitive_jvps[lox_p] = lox_jvp

def lox_p_transpose(ct, x):
    raise ValueError("Transpose doesn't support logging")
ad.primitive_transposes[lox_p] = lox_p_transpose
