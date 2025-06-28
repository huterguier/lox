import jax
import jax.numpy as jnp


class logdict(dict):
  """
  A dictionary that stores log values and the steps at which they were recorded.
  """

  def __init__(self, data, **kwargs):
    super().__init__(data)
    for key, value in kwargs.items():
      

      

