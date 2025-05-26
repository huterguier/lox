import jax
import jax.numpy as jnp
from jax.extend import core
from jax.interpreters import mlir, batching, ad
from jax._src.debugging import DebugEffect
from jax.interpreters.mlir import emit_python_callback


lox_p = core.Primitive("lox")
lox_p.multiple_results = True

def log(data, step=None):
    def callback(data):
        print("Logging data:", data)
    data_flat, structure = jax.tree.flatten(data)
    return lox_p.bind(*data_flat, structure=structure, callback=callback)

@lox_p.def_impl
def lox_impl(*data_flat, structure, callback):
    data = jax.tree.unflatten(structure, data_flat)
    callback(data)
    return data_flat

@lox_p.def_effectful_abstract_eval
def lox_abstract_eval(*data_flat, structure, callback):
    return list(data_flat), {DebugEffect()}

def lox_lowering(*data_flat, structure, callback):
    data = jax.tree.unflatten(structure, data_flat)
    jax.debug.callback(callback, data)
    return data_flat
mlir.register_lowering(lox_p, mlir.lower_fun(lox_lowering, multiple_results=True))

def lox_batch(vector_arg_values, batch_axes):
    outs = log(*vector_arg_values)
    return outs, (0,) * len(outs)
batching.primitive_batchers[lox_p] = lox_batch

def lox_jvp(arg_values, arg_tangents):
    return lox_p.bind(*arg_values), []
ad.primitive_jvps[lox_p] = lox_jvp

def lox_p_transpose(ct, x):
    raise ValueError("Transpose doesn't support logging")
ad.primitive_transposes[lox_p] = lox_p_transpose
