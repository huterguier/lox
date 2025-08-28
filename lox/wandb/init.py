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
    logs: lox.logdict,
):
    def callback(id, logs):
        id = str(lox.String(id))
        run = runs[id]
        if "step" in logs.steps:
            ordered_data = {}
            for k, vs in logs.items():
                steps = logs.steps["step"]
                if k not in steps:
                    raise ValueError(f"Either all or none of the keys must have steps. Key {k} is missing steps.")
                for v, step in zip(vs, steps[k]):
                    step = int(step)
                    if step not in ordered_data:
                        ordered_data[step] = {k: v}
                    else:
                        ordered_data[step] |= {k: v}
            for step in sorted(ordered_data.keys()):
                run.log(ordered_data[step], step=step)
        else:
            leaves = jax.tree.leaves(logs)
            assert all([len(leaf) == len(leaves[0]) for leaf in leaves])
            for i in range(len(leaves[0])):
                run.log(jax.tree.map(lambda l: l[i], logs))

    jax.debug.callback(
        callback, ordered=True, id=run.id, logs=logs
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
