# Quick Start

Welcome to the `lox` quick start guide! This page will walk you through the basic concepts and API of `lox`, helping you get up and running with logging in JAX in no time. We'll cover the core transformations, how to handle logs with `logdict`, and how to use built-in loggers.

## Basic API

At its core `lox` is built around 2 central function transformations calles `tap` and `spool`.
They work by traversing the functions [`jaxpr`](https://docs.jax.dev/en/latest/jaxpr.html), JAX's internal intermediate function representation, and dynamically alters it to match the desired behavior.
In order to use them with you function, all you need to do is specify what you want to log using `lox.log`.

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

## Logdicts

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

`lox.log` also accepts a `tags` argument, letting you mark individual log calls so they can be selectively kept or dropped later by `lox.tap`, `lox.spool`, and `lox.strip`, without changing the call site itself.

```python
>>> lox.log({"carry": carry}, tags=["debug"])
>>> _, logs = lox.spool(f, tags=["debug"])(xs)
```

## Loggers

Lox comes with built-in loggers for common use cases.
Loggers support both `lox.tap` and `lox.spool` transformations and let you easily log to different backends.
An example is `lox.loggers.SaveLogger`, which saves logs to a specified directory in a structured format for later use. Loggers are instantiaded with any necessary configuration, and then initialized with a random key using `init` to produce a logger state. This state is then passed to the `tap` or `spool` transformation along with the function to be logged.

```python
>>> import lox.loggers
>>> key = jax.random.key(0)
>>> logger = lox.loggers.SaveLogger("./.lox/")
>>> logger_state = logger.init(key)
>>> y = logger.spool(f, logger_state)(xs)
```

Loggers can also be combined to log to multiple backends simultaneously using `lox.loggers.MultiLogger`. The difference between `tap` and `spool` is preserved, so you can use `MultiLogger` with either transformation. Hence `spool` only logs once at the env of the function execution, while `tap` logs every time a log is encountered.

```python
>>> console_logger = lox.loggers.ConsoleLogger()
>>> save_logger = lox.loggers.SaveLogger("./.lox/")
>>> multi_logger = lox.loggers.MultiLogger(console_logger, save_logger)
>>> multi_logger_state = multi_logger.init(key)
>>> y = multi_logger.tap(f, multi_logger_state)(xs)
```
