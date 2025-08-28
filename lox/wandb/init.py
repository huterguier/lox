import jax
import jax.numpy as jnp
import jax.experimental
import wandb
from lox.wandb.run import Run
import lox
from .run import runs
from functools import partial
from typing import Optional


def log(
    run: Run,
    data: dict[str, jax.Array],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
):
    def callback(id, data, step, commit):
        id = str(lox.String(id))
        run = runs[id]
        run.log(data, step=step, commit=commit)

    jax.debug.callback(
        callback, ordered=True, id=run.id, data=data, step=step, commit=commit
    )
    return


def init(key, **kwargs):
    def callback(key, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, jax.Array):
                kwargs[k] = str(lox.String(v))
        if "name" in kwargs:
            kwargs["name"] = kwargs["name"] + f"{key[0]}{key[1]}"

        run = wandb.init(reinit="create_new", **kwargs)
        runs[run.id] = run

        return lox.string(run.id).value

    if "config" in kwargs:
        config = kwargs["config"]
        kwargs = {k: v for k, v in kwargs.items() if k != "config"}
        callback = partial(callback, config=config)

    for k, v in kwargs.items():
        if isinstance(v, str):
            kwargs[k] = lox.string(v).value
    id = jax.experimental.io_callback(
        callback,
        result_shape_dtypes=jax.ShapeDtypeStruct((lox.util.LENGTH,), jnp.uint8),
        key=key,
        **kwargs,
    )

    return Run(id=id)


def finish(run: Run):
    def callback(id):
        id = str(lox.String(id))
        run = runs[id]
        run.finish()

    jax.debug.callback(callback, ordered=True, id=run.id)
    return
