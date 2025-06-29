import jax
from jax.extend import core
from jax.interpreters import mlir, batching, ad
from jax._src.debugging import DebugEffect


lox_p = core.Primitive("lox")
lox_p.multiple_results = True

def log(data, **steps):
    data_flat, structure = jax.tree.flatten(data)
    return lox_p.bind(*data_flat, structure=structure)

@lox_p.def_impl
def lox_impl(*data_flat, structure):
    data = jax.tree.unflatten(structure, data_flat)
    return data_flat

@lox_p.def_effectful_abstract_eval
def lox_abstract_eval(*data_flat, structure):
    return list(data_flat), {DebugEffect()}

def lox_lowering(*data_flat, structure):
    data = jax.tree.unflatten(structure, data_flat)
    return data_flat
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
