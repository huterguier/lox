import jax
import jax.numpy as jnp
import jax.experimental
import wandb
import lox
from lox.wandb.run import Run
from rich.logging import RichHandler
import logging
import os

# os.environ["WANDB_SILENT"] = "true"
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
        run.log(data, **kwargs)
    jax.debug.callback(
        callback, 
        ordered=True, 
        id=run.id, 
        data=data
    )
    return


def init(key, *args, **kwargs):
    def callback(key):
        if wandb.run is not None:
            wandb.run.finish()
        run = wandb.init(*args, **kwargs)
        id = lox.string(run.id)
        print(f"\033[94mwandb(lox)\033[0m: Initializing wandb run with id: {id}")
        return id.value
    id = jax.experimental.io_callback(
        callback, 
        result_shape_dtypes=jax.ShapeDtypeStruct((lox.util.LENGTH,), jnp.uint8), 
        key=key)
    return Run(id=id)
