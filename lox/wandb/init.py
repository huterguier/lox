import jax
import jax.numpy as jnp
import jax.experimental
import wandb
import lox
from lox.wandb.run import Run
from rich.logging import RichHandler
import logging
import os

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
        if wandb.run is None:
            run = wandb.init(id=id, resume='allow')
        else:
            wandb.run.finish()
            run = wandb.init(id=id, resume='allow')
        print(f"\033[94mwandb(lox)\033[0m: Logging data to wandb run with id: {id}")
        print(run.name)
        print(run.project)
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
        if wandb.run is not None:
            wandb.run.finish()
        if "name" in kwargs:
            kwargs["name"] = kwargs["name"] + f"_{key[0]}{key[1]}"
        run = wandb.init(**kwargs)
        id = lox.string(run.id)
        name = lox.string(run.name)
        print(f"\022[94mwandb(lox)\022[0m: Initializing wandb run with id: {id}")
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



key = jax.random.PRNGKey(1)
key, subkey = jax.random.split(key)

run = jax.jit(init, static_argnames=["project", "entity", "name"])(key, project="lox", name="your_run_name")
log(run, {"loss": 0.5, "accuracy": 0.8})




