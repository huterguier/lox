import jax
import jax.numpy as jnp
import jax.experimental
import pickle
import os
from typing import Any

import lox


def save(data: dict[str, Any], path: lox.String, key: jax.Array = None):
  def _callback(data: dict[str, Any], path, key):
    path = str(lox.String(path))
    if key is not None:
      path = path + '/' + f"{key[0]}{key[1]}"

    for key in data:
      file = path + '/' + key + '.pkl'
      if os.path.exists(file):
        with open(file, 'rb') as f:
          value = pickle.load(f)
        value = jax.tree.map(lambda v, d: jnp.concatenate([v, d]), value, data[key])
        with open(file, 'wb') as f:
          pickle.dump(value, f)
      else:
        if not os.path.exists(path):
          os.makedirs(path)
        with open(file, 'wb') as f:
          pickle.dump(data[key], f)

  jax.debug.callback(
    _callback,
    ordered=True,
    data=data,
    path=path.value,
    key=key
  )

  return


# def load(path, result_shape_dtypes):
#
#   def _len(path):
#     with open(path, 'rb') as f:
#       data = pickle.load(f)
#     return len(jax.tree.leaves(data)[0])
#
#
#   def _callback(path):
#     with open(path, 'rb') as f:
#       data = pickle.load(f)
#     return data
#
#   data = jax.experimental.io_callback(
#     _callback,
#     result_shape_dtypes=result_shape_dtypes,
#     path=path
#   )
#
#   return data
#
#
# data = {"a": jax.numpy.ones((2, 3)), "b": jax.numpy.ones((2, 3))}
# path = lox.string('test_save_path')
# save(data, path)
