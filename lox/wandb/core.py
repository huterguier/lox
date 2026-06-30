from dataclasses import dataclass
from functools import partial

import jax
import jax.experimental
import wandb

from lox import logdict
from lox.utils.string_array import StringArray
from lox.utils.typing import Key

runs_wandb = {}


@jax.tree_util.register_dataclass
@dataclass
class WandbRun:
    """
    A class representing a run in Weights & Biases (wandb).
    This class is used to encapsulate the run ID. All other metadata is obtained by reconnecting to the existing run and retrieving the metadata from there.

    Attributes:
        id (Any): The unique identifier for the run in wandb.
    """

    id: StringArray

    def __init__(self, id: StringArray):
        """
        Initializes the Run instance with the given run ID.

        Args:
            id (Any): The unique identifier for the run in wandb.
        """
        self.id = id


def log(run: WandbRun, logs: logdict):
    """
    Logs data to the specified wandb run.
    Args:
        run (WandbRun): The WandbRun instance representing the run to log data to.
        logs (logdict): A logdict containing the data to be logged.
    """

    def flatten(data: dict):
        flat_data = {}
        for k, v in data.items():
            if isinstance(v, dict):
                nested_flat_data = flatten(v)
                for nk, nv in nested_flat_data.items():
                    flat_data[f"{k}/{nk}"] = nv
            else:
                flat_data[k] = v
        return flat_data

    def callback(id, logs):
        id = str(id)
        run = runs_wandb[id]
        if "step" in logs:
            step_arr = logs["step"]
            data_keys = [k for k in logs if k != "step"]
            ordered_data = {}
            for i, step in enumerate(step_arr):
                step = int(step)
                if step not in ordered_data:
                    ordered_data[step] = {}
                for k in data_keys:
                    ordered_data[step][k] = logs[k][i]
            for step in sorted(ordered_data.keys()):
                data = flatten(ordered_data[step])
                run.log(data, step=step)
        else:
            leaves = jax.tree.leaves(logs)
            if leaves:
                assert all([len(leaf) == len(leaves[0]) for leaf in leaves])
                for i in range(len(leaves[0])):
                    data = flatten(jax.tree.map(lambda l: l[i], logs))
                    run.log(data)

    jax.debug.callback(callback, ordered=True, id=run.id, logs=logs)
    return


def init(key: Key, **kwargs):
    """
    Initializes a new wandb run with the provided keyword arguments.
    Args:
        key (Key): A random key used to initialize the run.
        **kwargs: Additional keyword arguments to be passed to wandb.init().
    Returns:
        WandbRun: An instance of WandbRun representing the initialized run.
    """

    def callback(key, **kwargs):
        if "name" in kwargs:
            key_data = jax.random.key_data(key)
            seed = str(int(f"{key_data[0]}{key_data[1]}"))
            kwargs["name"] = kwargs["name"] + seed

        run = wandb.init(reinit="create_new", **kwargs)
        runs_wandb[run.id] = run

        return StringArray.from_str(run.id, length=64)

    result_shape_dtypes = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        StringArray.from_str("", length=64),
    )
    id = jax.experimental.io_callback(
        partial(callback, **kwargs),
        result_shape_dtypes=result_shape_dtypes,
        key=key,
    )

    return WandbRun(id=id)


def finish(run: WandbRun):
    """
    Finishes the specified wandb run.
    Args:
        run (WandbRun): The WandbRun instance representing the run to be finished.
    """

    def callback(id):
        id = str(id)
        run = runs_wandb[id]
        run.finish()

    jax.debug.callback(callback, ordered=True, id=run.id)
    return
