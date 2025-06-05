import jax
from dataclasses import dataclass
from typing import Any


@jax.tree_util.register_dataclass
@dataclass
class Run:
    id: Any
