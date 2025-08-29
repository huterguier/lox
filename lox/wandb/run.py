import jax
from dataclasses import dataclass
from typing import Any

runs_wandb = {}


@jax.tree_util.register_dataclass
@dataclass
class WandbRun:
    """
    A class representing a run in Weights & Biases (wandb).
    This class is used to encapsulate the run ID. All other metadata is obtained by reconnecting to the existing run and retrieving the metadata from there.

    Attributes:
        id (Any): The unique identifier for the run in wandb.
    """

    id: Any

    def __init__(self, id: Any):
        """
        Initializes the Run instance with the given run ID.

        Args:
            id (Any): The unique identifier for the run in wandb.
        """
        self.id = id

    def log(
        self, data: dict[str, Any], step: int | None = None, commit: bool | None = None
    ) -> None:
        """
        Logs data to the run.
        Wraps :func:`lox.wandb.log` to log data to the run.

        Args:
            data (dict): The data to log to the run.
        """
        pass
