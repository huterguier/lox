<div align="center">
    <img src="https://github.com/huterguier/lox/blob/main/docs/_static/lox.png" width="230">
</div>

# Accelerated logging in JAX

Lox is a lightweight and flexible logging library for JAX that provides a simple interface for logging data during function execution.
Logging is implemented with it's own primitive, which allows it to work seamlessly with JAX's built-in function transformations like `jit` or `vmap`.
All you need to do is decorate your code with `lox.log` statements and Lox does the rest.
Using JAX's intermediate function representation Lox can dynamically insert callbacks to log you data or collect the logs that would have been generated during the execution and return them as part of the output of you function.
While it's obviously possible to implement this functionality yourself, Lox provides a simple and efficient way to do so without having to carry around boilerplate code in your functions.

```python
def f(xs):
    lox.log({"xs": xs})
    def scan_step(carry, x):
        carry += x
        lox.log({"carry": carry})
        return carry, x
    y, _ = jax.lax.scan(scan_step, 0, xs)
    return ys

xs = jnp.arange(5)
```
The first transformation, `lox.tap`, lets you "tap into" function execution by attaching a callback that receives logs as they're generated. It streams logs in real time, making it great for debugging or live monitoring.

```python
def callback(logs):
    print("Logging: ", logs)
y = lox.tap(f, callback=callback)(xs)
>>> Logging: {"xs": [0, 1, 2]}
>>> Logging: {"carry": 0}
>>> Logging: {"carry": 1}
>>> Logging: {"carry": 2}
```

The second transformation, `lox.spool`, "spools up" all logs during execution and returns them alongside the function's output. 
This is especially useful when frequent callbacks would be too expensive. 
For instance, instead of logging on every iteration, you can collect all logs for a training step and emit them in a single call.
```python
y, logs = lox.spool(f)(xs)
print("Collected Logs: ", logs)
>>>> Collected Logs: { "xs": [0, 1, 2], "carry": [[0, 1, 2]] }
```

## Installation

Lox can be installed via pip directly from the GitHub repository.

```bash
pip install git+https://github.com/huterguier/lox
```
