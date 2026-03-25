from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from lox.loggers.console_logger import ConsoleLogger
from lox.loggers.logger import Logger
from lox.loggers.multi_logger import MultiLogger
from lox.loggers.save_logger import SaveLogger

if TYPE_CHECKING:
    from lox.matplotlib.matplotlib_logger import MatplotlibLogger
    from lox.wandb.wandb_logger import WandbLogger

_OPTIONAL_LOGGERS = {
    "WandbLogger": ("lox.wandb.wandb_logger", "wandb"),
    "MatplotlibLogger": ("lox.matplotlib.matplotlib_logger", "matplotlib"),
}

__all__ = [
    "ConsoleLogger",
    "Logger",
    "MultiLogger",
    "SaveLogger",
    "WandbLogger",
    "MatplotlibLogger",
]


def __getattr__(name: str) -> Any:
    if name in _OPTIONAL_LOGGERS:
        module_path, package_name = _OPTIONAL_LOGGERS[name]
        try:
            module = importlib.import_module(module_path)
            return getattr(module, name)
        except ImportError as e:
            # Check if the missing dependency is the package itself or something else
            try:
                importlib.import_module(package_name)
            except ImportError:
                from lox.utils.optional import MissingDependency

                return MissingDependency(name, package_name)
            raise e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__ + list(globals().keys()))
