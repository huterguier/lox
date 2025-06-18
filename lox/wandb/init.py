import jax
import jax.numpy as jnp
import jax.experimental
import wandb
from lox.wandb.run import Run
import lox
from .runs import runs
from functools import partial


def is_hashable(arg):
  """
  Check if an argument is hashable.

  Args:
      arg: The argument to check.
  Returns:
      bool: True if the argument is hashable, False otherwise.
  """
  try:
    hash(arg)
    return True
  except TypeError:
    return False


def log(run: Run, data, **kwargs):
    def callback(id, data):
        id = str(lox.String(id))
        run = runs[id]
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

        return lox.string(run.id).value

    static_kwargs = {k: v for k, v in kwargs.items() if is_hashable(v)}
    dynamic_kwargs = {k: v for k, v in kwargs.items() if not is_hashable(v)}

    for k, v in dynamic_kwargs.items():
        if isinstance(v, str):
            kwargs[k] = lox.string(v).value

    id = jax.experimental.io_callback(
        partial(callback, **static_kwargs),
        result_shape_dtypes=jax.ShapeDtypeStruct((lox.util.LENGTH,), jnp.uint8),
        key=key,
        **kwargs
    )

    return Run(id=id)


def finish(run: Run):
    def callback(id):
        id = str(lox.String(id))
        run = runs[id]
        run.finish()

    jax.debug.callback(
        callback, 
        ordered=True, 
        id=run.id
    )
    return
