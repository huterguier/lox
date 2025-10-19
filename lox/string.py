from dataclasses import dataclass

import jax
import jax.numpy as jnp

from lox.typing import Array


@jax.tree_util.register_dataclass
@dataclass
class StringArray:
    chars: Array
    padding: Array

    def __init__(self, chars: Array, padding: Array = jnp.uint8(255)):
        self.chars = chars
        self.padding = padding

    def __str__(self):
        return "".join(chr(c) for c in self.chars if c != self.padding)

    def __getitem__(self, key):
        return self.chars[key]

    def __repr__(self):
        return f"StringArray({self.chars})"

    def __len__(self):
        return len(self.chars)

    @property
    def length(self):
        idx = jnp.where(self.chars == self.padding, size=1)[0][0]
        return idx

    def append(self, other: "StringArray") -> "StringArray":
        idx = jnp.where(self.chars == self.padding, size=1)[0][0]
        mask = jnp.arange(len(self.chars)) >= idx
        new_chars = jnp.where(mask, other.chars, self.chars)
        return StringArray(new_chars, self.padding)

    @classmethod
    def from_str(
        cls, s: str, length: int = 1024, padding: Array = jnp.uint8(255)
    ) -> "StringArray":
        chars = jnp.full((length,), padding, dtype=jax.numpy.uint8)
        for i, c in enumerate(s):
            if i >= length:
                raise ValueError("String too long")
            chars = chars.at[i].set(ord(c))
        return cls(chars, padding)
