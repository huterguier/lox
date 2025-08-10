# Lox <small><em>(Logs)</em></small>

Lox is a logging library for [JAX](https://docs.jax.dev/en/latest/) that provides a simple and flexible way to log metrics, and other data during the execution of pure functions.
It is designed to be easy to use and integrate with existing JAX workflows, while also providing powerful features for advanced logging needs.


## Features

- **Simple API**: Lox provides a simple and intuitive API for logging.
- **Flexible**: Lox is designed to be flexible and can be easily integrated into existing projects.
- **Jit-Compatible**: Lox is fully compatible with JAX's function transformations, such as `jit`, `vmap`, and `pmap`.
- **Advanced Features**: Lox provides advanced features such as logging to multiple backends, custom loggers, and more.


## Installation

Lox is not yet available on PyPI, but it can be installed directly from GitHub using pip.
```bash
pip install git+https://github.com/huterguier/lox
```

```{toctree}
:hidden:
:maxdepth: 2

quick_start
the_sharp_bits
examples
api
```
