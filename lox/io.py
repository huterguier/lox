import os
import pickle
from collections.abc import Iterable
from functools import partial
from typing import Any

import jax
import jax.experimental
import jax.numpy as jnp

from lox.utils import get_path
from lox.utils.string_array import StringArray
from lox.utils.typing import Key


def save_callback(
    data: dict[str, Any],
    path: StringArray | str,
    mode: str = "a",
    key: Key | None = None,
):
    def append(file, v):
        with open(file, "rb") as f:
            v_file = pickle.load(f)
        v_file = jax.tree.map(
            lambda vf, vd: jnp.concatenate([jnp.atleast_1d(vf), jnp.atleast_1d(vd)]),
            v_file,
            v,
        )
        with open(file, "wb") as f:
            pickle.dump(v_file, f)

    def write(file, v):
        with open(file, "wb") as f:
            pickle.dump(v, f)

    def exclusive(file, v):
        if os.path.exists(file):
            raise FileExistsError(f"File {file} already exists.")
        with open(file, "wb") as f:
            pickle.dump(v, f)

    def save_data(path, data):
        os.makedirs(path, exist_ok=True)
        for k, v in data.items():
            file = os.path.join(path, f"{k}.pkl")
            dir, _ = os.path.split(file)
            if not os.path.exists(dir):
                os.makedirs(dir)
            if mode == "a":
                if os.path.exists(file):
                    append(file, v)
                else:
                    write(file, v)
            elif mode == "w":
                write(file, v)
            elif mode == "x":
                exclusive(file, v)

    if mode not in ["a", "w", "x"]:
        raise ValueError("Mode must be 'a', 'w', or 'x'.")
    if isinstance(path, StringArray):
        path = str(path)
    if key is not None:
        if len(key.shape) >= 1:
            keys = key.flatten()
            data_flat = jax.tree.map(
                lambda x: x.reshape(len(keys), *x.shape[len(key.shape) :]), data
            )
            leaves, treedef = jax.tree.flatten(data_flat)
            datas = [
                treedef.unflatten(leaf_tuple)
                for leaf_tuple in zip(*leaves, strict=True)
            ]
            for i, data in enumerate(datas):
                path_i = get_path(path, keys[i])
                save_data(path_i, data)
        else:
            path = get_path(path, key)
            save_data(path, data)
    else:
        save_data(path, data)


def save(
    data: dict[str, Any],
    path: StringArray | str,
    mode: str = "a",
    key: jax.Array | None = None,
):
    """
    Save data to a specified path using a callback function. Each entry in the data dictionary is saved as a separate file with the key as the filename.

    Args:
      data (dict[str, Any]): The data to be saved.
      path (StringArray): The path where the data will be saved.
      mode (str): The mode in which to open the file ('a' for append, 'w' for write, 'x' for exclusive creation).
      key (jax.Array, optional): An optional key to differentiate data when saving.
    """
    callback = partial(save_callback, path=path, mode=mode)
    jax.debug.callback(callback, data=data, key=key)


def load_callback(
    path: StringArray | str,
    argnames: str | Iterable[str] | None = None,
    key: Key | None = None,
) -> dict[str, Any]:
    if isinstance(argnames, str):
        argnames = (argnames,)

    def load_data(path):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"No such directory: {path!r}")
        data = {}
        if argnames is None:
            for root, _, files in os.walk(path):
                dir = os.path.relpath(root, path)
                for file in files:
                    filename = os.path.normpath(os.path.join(dir, file))
                    if filename.endswith(".pkl"):
                        argname = filename[:-4]
                        file_path = os.path.join(path, filename)
                        with open(file_path, "rb") as f:
                            data[argname] = pickle.load(f)
        else:
            for argname in argnames:
                file_path = os.path.join(path, f"{argname}.pkl")
                with open(file_path, "rb") as f:
                    data[argname] = pickle.load(f)
        return data

    if isinstance(path, StringArray):
        path = str(path)
    if key is None:
        return load_data(path)
    else:
        if key.ndim >= 1:  # or len(key.shape) > 1
            keys = key.reshape(-1)
            datas = []
            for k in keys:
                path_k = get_path(path, k)
                data_k = load_data(path_k)
                datas.append(data_k)
            data = jax.tree.map(lambda *d: jnp.stack(d), *datas)
            data = jax.tree.map(lambda x: x.reshape(key.shape + x.shape[1:]), data)
            return data
        else:
            path_key = get_path(path, key)
            return load_data(path_key)


def load(
    path: StringArray | str,
    key: Key | None = None,
    result_shape_dtypes: Any = None,
    argnames: str | Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    Load data from a specified path. Each file in the directory is loaded into a dictionary with the filename (without extension) as the key.

    Args:
      path (StringArray): The path from which the data will be loaded.
      result_shape_dtypes (Any, optional): The expected shape and dtype of the loaded data.
      argnames (str | Iterable[str], optional): Specific argument name(s) to load. If None, all files in the directory are loaded.
      key (jax.Array, optional): An optional key to differentiate data when loading.
    Returns:
        dict[str, Any]: The loaded data.
    """
    if result_shape_dtypes is None:
        logs = load_callback(path=path, argnames=argnames, key=key)
    else:
        callback = partial(
            load_callback,
            path=path,
            argnames=argnames,
        )
        logs = jax.experimental.io_callback(
            callback,
            result_shape_dtypes,
            key=key,
        )
    return logs
