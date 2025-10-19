import os
import pickle
from typing import Any, Optional

import jax
import jax.numpy as jnp

import lox


def save_callback(
    data: dict[str, Any],
    path: lox.String | str,
    mode: str = "a",
    key: Optional[jax.Array] = None,
):
    if mode not in ["a", "w", "x"]:
        raise ValueError("Mode must be 'a', 'w', or 'x'.")
    if isinstance(path, lox.String):
        path = str(lox.String(path))
    if key is not None:
        key_data = jax.random.key_data(key)
        folder_name = str(int(f"{key_data[0]}{key_data[1]}"))
        path = path + "/" + folder_name

    for k, v in data.items():
        file = path + f"/{k}.pkl"
        dir, _ = os.path.split(file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        if mode == "a":
            if os.path.exists(file):
                with open(file, "rb") as f:
                    v_file = pickle.load(f)
                v_file = jax.tree.map(
                    lambda vf, vd: jnp.concatenate(
                        [jnp.atleast_1d(vf), jnp.atleast_1d(vd)]
                    ),
                    v_file,
                    v,
                )
                with open(file, "wb") as f:
                    pickle.dump(v_file, f)
            else:
                with open(file, "wb") as f:
                    pickle.dump(v, f)
        elif mode == "w":
            with open(file, "wb") as f:
                pickle.dump(v, f)
        elif mode == "x":
            if os.path.exists(file):
                raise FileExistsError(f"File {file} already exists.")
            with open(file, "wb") as f:
                pickle.dump(v, f)


def save(
    data: dict[str, Any],
    path: lox.String | str,
    mode: str = "a",
    key: Optional[jax.Array] = None,
):
    """
    Save data to a specified path using a callback function. Each entry in the data dictionary is saved as a separate file with the key as the filename.
    Args:
      data (dict[str, Any]): The data to be saved.
      path (lox.String): The path where the data will be saved.
      mode (str): The mode in which to open the file ('a' for append, 'w' for write, 'x' for exclusive creation).
      key (jax.Array, optional): An optional key to differentiate data when saving.
    """
    jax.debug.callback(
        save_callback, ordered=True, data=data, path=path, mode=mode, key=key
    )


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
