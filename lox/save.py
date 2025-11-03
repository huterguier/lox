import os
import pickle
from typing import Any, Iterable, Optional

import jax
import jax.experimental
import jax.numpy as jnp

from lox.string import StringArray
from lox.typing import Key


def save_callback(
    data: dict[str, Any],
    path: StringArray | str,
    mode: str = "a",
    key: Optional[Key] = None,
):
    if mode not in ["a", "w", "x"]:
        raise ValueError("Mode must be 'a', 'w', or 'x'.")
    if isinstance(path, StringArray):
        path = str(path)
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
    path: StringArray | str,
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


def load_callback(
    path: StringArray | str,
    result_shape_dtypes: Any,
    argnames: Iterable[str],
    key: Optional[Key] = None,
) -> dict[str, Any]:
    def callback(key, path):
        if key is not None:
            key_data = jax.random.key_data(key)
            folder_name = str(int(f"{key_data[0]}{key_data[1]}"))
            path = path + "/" + folder_name
        data = {}
        for argname in argnames:
            file_path = path + f"/{argname}.pkl"
            with open(file_path, "rb") as f:
                data[argname] = pickle.load(f)
        return data

    return jax.experimental.io_callback(
        callback,
        result_shape_dtypes,
        path=path,
        key=key,
    )


def load(
    path: StringArray | str,
    result_shape_dtypes: Any = None,
    key: Optional[Key] = None,
) -> dict[str, Any]:
    """
    Load data from a specified path. Each file in the directory is loaded into a dictionary with the filename (without extension) as the key.
    Args:
      path (lox.String): The path from which the data will be loaded.
      key (jax.Array, optional): An optional key to differentiate data when loading.
    Returns:
        dict[str, Any]: The loaded data.
    """
    raise NotImplementedError("Load function is not implemented yet.")
