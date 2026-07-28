<div align="center">
    <img src="https://github.com/huterguier/lox/blob/main/docs/_static/lox.png" width="230">
</div>

# Logging in JAX
[![PyPI version](https://img.shields.io/badge/pypi-not_available-red.svg)](#installation)
[![License: Apache-2.0](https://img.shields.io/github/license/huterguier/lox?color=yellow)](https://www.apache.org/licenses/LICENSE-2.0)
[![Documentation](https://img.shields.io/badge/docs-available-blue.svg)](https://jax-lox.readthedocs.io/en/latest/)
[![Code Style: Ruff](https://img.shields.io/badge/codestyle-ruff-261230.svg)](https://docs.astral.sh/ruff/formatter/)


[`lox`](https://github.com/huterguier/lox) is a lightweight and flexible logging library for [JAX](https://github.com/jax-ml/jax).
All you need to do is decorate your code with `lox.log` statements and `lox` does the rest.
Using JAX's intermediate function representation Lox can dynamically insert callbacks to log your data, or collect the logs that would have been generated during the execution and return them as part of the output of your function.
While it's obviously possible to implement this functionality yourself, `lox` provides a simple and efficient way to do so without having to carry around boilerplate code in your functions.

## Features
- 🔌 **Plug-and-Play:**  Simply add `lox.log` statements where you need them. `lox` handles all the complex boilerplate of plumbing data through JAX's transformations, keeping your function signatures clean and focused on the logic.

- 📦 **Automatic Extraction:**  Instead of explicitly returning data from your functions, `lox.spool` automatically "spools up" all logs generated during a function's execution. It collects them and returns them as a single `logdict` alongside the function's original output.

- 📡 **Dynamic Callbacks:**  Using `lox.tap`, you can "tap into" a JAX-transformed function using custom callbacks. This is ideal for live monitoring and debugging without halting execution.

- ✅ **`vmap` over Seeds:**  Built on its own JAX primitive, `lox` works effortlessly with core transformations like `jit`, `scan`, and `vmap`.

- 📊 **Experiment Loggers:**  Includes built-in loggers that seamlessly pipe your metrics to popular experiment tracking platforms including [`wandb`](https://wandb.ai/) and [`neptune`](https://neptune.ai/), which are also fully compatible with `vmap`.

## Quick Start

### Basic API
At its core `lox` is built around 2 central function transformations called `tap` and `spool`.
They work by traversing the function's [`jaxpr`](https://docs.jax.dev/en/latest/jaxpr.html), JAX's internal intermediate function representation, and dynamically alters it to match the desired behavior.
In order to use them with your function, all you need to do is specify what you want to log using `lox.log`.

```python
>>> import jax
>>> import jax.numpy as jnp
>>> import lox

>>> def f(xs):
...     lox.log({"xs": xs})
...     def step(carry, x):
...         carry += x
...         lox.log({"carry": carry})
...         return carry, x
...     y, _ = jax.lax.scan(step, 0, xs)
...     return y

>>> xs = jnp.arange(3)
```
The first transformation, `lox.tap`, lets you "tap into" function execution by attaching a callback that receives logs as they're generated. 
It streams logs in real time, making it great for debugging or live monitoring.
In the following example we use a simple callback that writes all logs to the console.

```python
>>> def callback(logs):
...     print("Logging:", logs)
>>> y = lox.tap(f, callback=callback)(xs)
Logging: {'xs': [0, 1, 2]}
Logging: {'carry': 0}
Logging: {'carry': 1}
Logging: {'carry': 3}
```

The second transformation, `lox.spool`, "spools up" all logs during execution and returns them alongside the function's output. 
This is especially useful when frequent callbacks would be too expensive. 
For instance, instead of logging on every iteration, you can collect all logs for a training step and emit them in a single call.
`spool` is also particularly useful for collecting logs over multiple steps and then applying a reduction like `jnp.mean` to them.
```python
>>> y, logs = lox.spool(f)(xs)
>>> print("Collected Logs:", logs)
Collected Logs: {'xs': [0, 1, 2], 'carry': [0, 1, 3]}
```

### Logdicts

Lox provides its own internal data structure for logs called `logdict`, which is a subclass of Python's built-in `dict`.
To the naked eye, it behaves like a regular dictionary, but it comes with some additional features that make it easier to work with logs.

```python
>>> def f(xs):
...     def body(i, carry):
...         carry += xs[i]
...         lox.log({"carry": carry})
...         return carry
...     y = jax.lax.fori_loop(0, len(xs), body, 0)
...     return y
>>> y, logs = lox.spool(f)(xs)
>>> print("Collected Logs:", logs["carry"])
Collected Logs: [0, 1, 3]
```

`lox.log` also accepts a `tags` argument, letting you mark individual log calls so they can be selectively kept or dropped later by `lox.tap`, `lox.spool`, `lox.strip`, and `lox.keep`, without changing the call site itself.

```python
>>> lox.log({"carry": carry}, tags=["debug"])
>>> _, logs = lox.spool(f, tags=["debug"])(xs)
```

### Loggers

Lox comes with built-in loggers for common use cases.
Loggers support both `lox.tap` and `lox.spool` transformations and let you easily log to different backends.
An example is `lox.loggers.SaveLogger`, which saves logs to a specified directory in a structured format for later use. Loggers are instantiated with any necessary configuration, and then initialized with a random key using `init` to produce a logger state. This state is then passed to the `tap` or `spool` transformation along with the function to be logged.

```python
>>> import lox.loggers
>>> key = jax.random.key(0)
>>> logger = lox.loggers.SaveLogger("./.lox/")
>>> logger_state = logger.init(key)
>>> y = logger.spool(f, logger_state)(xs)
```

Loggers can also be combined to log to multiple backends simultaneously using `lox.loggers.MultiLogger`. The difference between `tap` and `spool` is preserved, so you can use `MultiLogger` with either transformation. Hence `spool` only logs once at the end of the function execution, while `tap` logs every time a log is encountered.

```python
>>> from lox.loggers import ConsoleLogger, MultiLogger, SaveLogger
>>> console_logger = ConsoleLogger()
>>> save_logger = SaveLogger(".lox/")
>>> multi_logger = MultiLogger(console_logger, save_logger)
>>> multi_logger_state = multi_logger.init(key)
>>> y = multi_logger.tap(f, multi_logger_state)(xs)
```


## Installation
`lox` can be installed directly from this GitHub repository.
```bash
pip install git+https://github.com/huterguier/lox
```
By default `lox` comes without any of the external experiment loggers. Make sure to include the optional dependencies or to install them manually.
```bash
pip install "lox[wandb,neptune] @ git+https://github.com/huterguier/lox"
```
