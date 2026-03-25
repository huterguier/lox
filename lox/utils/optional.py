from typing import Any


class MissingDependency:
    """A placeholder for classes that require an optional dependency."""

    def __init__(self, name: str, package_name: str):
        self._name = name
        self._package_name = package_name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            f"'{self._name}' requires the '{self._package_name}' package. "
            f"Please install it with `pip install lox[{self._package_name}]`."
        )

    def __getattr__(self, name: str) -> Any:
        # Allow attribute access to also raise the same ImportError
        return self.__call__
