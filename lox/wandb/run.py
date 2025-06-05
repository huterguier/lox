import jax
from dataclasses import dataclass
import lox


@jax.tree_util.register_dataclass
@dataclass
class Run:
    id: lox.String
    name: lox.String
