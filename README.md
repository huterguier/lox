<div align="center">
    <img src="https://github.com/huterguier/lox/blob/main/docs/_static/lox.png" width="230">
</div>

# Accelerated Logging in JAX

Lox is a lightweight and flexible logging library for JAX that provides a simple interface for logging data during function execution.
Logging is implemented with it's own primitive, which allows it to work seamlessly with JAX's built-in function transformations like `jit` or `vmap`.
All you need to do is decorate your code with `lox.log` statements and Lox does the rest.
Using JAX's intermediate function representation Lox can dynamically insert callbacks to log you data or collect the logs that would have been generated during the execution and return them as part of the output of you function.
While it's obviously possible to implement this functionality yourself, Lox provides a simple and efficient way to do so without having to carry around boilerplate code in your functions.

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
The first transformation, `lox.tap`, lets you "tap into" function execution by attaching a callback that receives logs as they're generated. It streams logs in real time, making it great for debugging or live monitoring.

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
```python
>>> y, logs = lox.spool(f)(xs)
>>> print("Collected Logs:", logs)
Collected Logs: {'xs': [0, 1, 2], 'carry': [0, 1, 3]}
```

## Logdicts

Lox provides its own internal data structure for logs called `logdict`, which is a subclass of Python's built-in `dict`.
To the naked eye, it behaves like a regular dictionary, but it comes with some additional features that make it easier to work with logs.
In addition to the raw data, a `logdict` also contains the steps at which the logs were recorded.
The following example demonstrates how to log data along with additional step information.

```python
>>> def f(xs):
...     def body(i, carry):
...         carry += xs[i]
...         lox.log({"carry": carry}, step=i, episode=i//2)
...         return carry
...     y = jax.lax.fori_loop(0, len(xs), body, 0)
...     return y
>>> y, logs = lox.spool(f)(xs)
```

In the example above, we log the `carry` value at each iteration of a loop, along with the current step and episode.
The step information can be accessed using attributes of the logdict.
We can then access them using `logs.step` and `logs.episode`.
An arbitrary amount of keywords can be added to `lox.log` which will all be treated as additional step information.

```python
>>> print("Collected Logs:", logs["carry"])
Collected Logs: [0, 1, 3]
>>> print("Corresponding Steps:", logs.step['carry'])
Corresponding Steps: [0, 1, 2]
>>> print("Corresponding Episodes:", logs.episode['carry'])
Corresponding Episodes: [0, 0, 1]
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





## Installation

Lox can be installed directly from the GitHub repository.

```bash
pip install git+https://github.com/huterguier/lox
```
