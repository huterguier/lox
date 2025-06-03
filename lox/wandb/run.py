import jax
import jax.numpy as jnp
import wandb
import base64
import struct
from dataclasses import dataclass


ENCODE_VECTOR_SIZE = 16

@jax.tree_util.register_dataclass
@dataclass
class Run:
    id: jnp.ndarray  # fixed-size id of uint8

    def decode(self) -> str:
        byte_repr = bytes(self.id.tolist())
        b64_encoded = base64.urlsafe_b64encode(byte_repr).decode('utf-8')
        return b64_encoded.replace('A', '').replace('=', '')

    @classmethod
    def encode(cls, encoded: str):
        byte_data = base64.urlsafe_b64decode(encoded)

        if len(byte_data) > ENCODE_VECTOR_SIZE:
            raise ValueError(f"Decoded data must be at most {ENCODE_VECTOR_SIZE} bytes")

        padded = byte_data.ljust(ENCODE_VECTOR_SIZE, b'\x00')
        vec = jnp.array(list(padded), dtype=jnp.uint8)
        return cls(vec)



