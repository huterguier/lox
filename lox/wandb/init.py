import jax
import jax.numpy as jnp
import jax.experimental
import wandb
from lox.wandb.run import Run
from rich.logging import RichHandler
import logging
import os
import lox
from .runs import runs


os.environ["WANDB_SILENT"] = "false"
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     handlers=[RichHandler()]
# )
# logger = logging.getLogger("lox.wandb")

def log(run: Run, data, **kwargs):
    def callback(id, data):
        id = str(lox.String(id))
        run = runs[id]
        print(f"\033[94mwandb(lox)\033[0m: Logging data to wandb run with id: {id}")
        run.log(data, **kwargs)

    jax.debug.callback(
        callback, 
        ordered=True, 
        id=run.id, 
        data=data
    )
    return


def init(key, **kwargs):
    def callback(key, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, jax.Array):
                kwargs[k] = str(lox.String(v))
        kwargs["name"] = kwargs["name"] + f"{key[0]}{key[1]}"

        run = wandb.init(reinit="create_new", **kwargs)
        runs[run.id] = run

        id = lox.string(run.id)
        name = lox.string(run.name)
        print(f"\033[94mwandb(lox)\033[0m: Initializing wandb run with id: {id}")
        return id.value, name.value

    for k, v in kwargs.items():
        if isinstance(v, str):
            kwargs[k] = lox.string(v).value
    id, name = jax.experimental.io_callback(
        callback, 
        result_shape_dtypes=2*(jax.ShapeDtypeStruct((lox.util.LENGTH,), jnp.uint8),),
        key=key,
        **kwargs
    )

    return Run(id=id)


def finish(run: Run):
    def callback(id):
        id = str(lox.String(id))
        run = runs[id]
        print(f"\033[94mwandb(lox)\033[0m: Finishing wandb run with id: {id}")
        run.finish()

    jax.debug.callback(
        callback, 
        ordered=True, 
        id=run.id
    )
    return
#
#
# key = jax.random.PRNGKey(1)
# key, subkey = jax.random.split(key)
#
# run = jax.jit(init, static_argnames=["project", "entity", "name"])(key, project="lox", name="your_run_name")
# log(run, {"loss": 0.5, "accuracy": 0.8})
# finish(run)




