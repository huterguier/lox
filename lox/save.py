import jax
import jax.numpy as jnp
import jax.experimental
import pickle
import os
from typing import Any
from lox.logger import Logger, LoggerState
from dataclasses import dataclass

import lox


def save(data: dict[str, Any], path: lox.String, mode: str='a', key: jax.Array = None):
  """
  Save data to a specified path using a callback function. Each entry in the data dictionary is saved as a separate file with the key as the filename.
  Args:
    data (dict[str, Any]): The data to be saved.
    path (lox.String): The path where the data will be saved.
    mode (str): The mode in which to open the file ('a' for append, 'w' for write, 'x' for exclusive creation).
    key (jax.Array, optional): An optional key to differentiate data when saving.
  """
  def callback(data: dict[str, Any], path, mode, key):
    if mode not in ['a', 'w', 'x']:
      raise ValueError("Mode must be 'a', 'w', or 'x'.")
    path = str(lox.String(path))
    if key is not None:
      path = path + '/' + f"{key[0]}{key[1]}"

    if mode == 'a':
      for key in data:
        file = path + '/' + key + '.pkl'
        if os.path.exists(file):
          with open(file, 'rb') as f:
            value = pickle.load(f)
          value = jax.tree_util.tree_map(
            lambda v, d: jnp.concatenate([jnp.atleast_1d(v), jnp.atleast_1d(d)]),
            value,
            data[key]
          )
          with open(file, 'wb') as f:
            pickle.dump(value, f)
        else:
          if not os.path.exists(path):
            os.makedirs(path)
          with open(file, 'wb') as f:
            pickle.dump(data[key], f)

    elif mode == 'w':
      for key in data:
        file = path + '/' + key + '.pkl'
        if not os.path.exists(path):
          os.makedirs(path)
        with open(file, 'wb') as f:
          pickle.dump(data[key], f)

    elif mode == 'x':
      if os.path.exists(path):
        raise FileExistsError(f"Path {path} already exists.")
      for key in data:
        file = path + '/' + key + '.pkl'
        if not os.path.exists(path):
          os.makedirs(path)
        with open(file, 'wb') as f:
          pickle.dump(data[key], f)

  jax.debug.callback(
    callback,
    ordered=True,
    data=data,
    path=path.value,
    mode=mode,
    key=key
  )


@dataclass
class SaveLoggerState(LoggerState):
  path: lox.String
  key: jax.Array


class SaveLogger(Logger[SaveLoggerState]):
  """
  Logger for saving data to a specified path using JAX's experimental IO callback.
  """

  def __init__(self, root: str):
    self.root = root

  def init(self, path: lox.String, key: jax.Array) -> SaveLoggerState:
    return SaveLoggerState(
      path=lox.String(self.root + '/' + str(path)),
      key=key
    )

  def log(self, logger_state: SaveLoggerState, logs: lox.logdict) -> None:
    save(logs, logger_state.path, key=logger_state.key)

  def tap(self, logger_state: SaveLoggerState, f: Any) -> Any:
    return lox.tap(f)(logger_state.path)

  def close(self, logger_state: SaveLoggerState):
    pass


# def load(path, result_shape_dtypes):
#
#   def _len(path):
#     with open(path, 'rb') as f:
#       data = pickle.load(f)
#     return len(jax.tree_util.tree_leaves(data)[0])
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
