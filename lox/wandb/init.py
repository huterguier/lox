import jax
import jax.numpy as jnp
import jax.experimental
import wandb


def wandb_log(key, *args, **kwargs):


def init(key, *args, **kwargs):
    def callback(key):
        print(jax.process_index(), "init called")
        run = wandb.init(*args, **kwargs)
        run.finish()
        return 0
    print(key)
    id = jax.experimental.io_callback(callback, result_shape_dtypes=jax.ShapeDtypeStruct((), jnp.int32), key=key)
    jax.debug.print("{id}", id=id)
    return id


key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 4)
print(keys)

id = jax.vmap(init, (0,))(keys)


# define datatype lox.sting(s: str) -> jnp.ndarray:
def lox_string(s: str) -> jax.Array:
    """Convert a string to a fixed-size JAX array of uint8."""
    byte_data = s.encode('utf-8')
    if len(byte_data) > Run.ENCODE_VECTOR_SIZE:
        raise ValueError(f"String must be at most {Run.ENCODE_VECTOR_SIZE} bytes")
    
    padded = byte_data.ljust(Run.ENCODE_VECTOR_SIZE, b'\x00')
    return jax.numpy.array(list(padded), dtype=jax.numpy.uint8)
