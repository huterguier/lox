<div align="center">
    <img src="https://github.com/huterguier/lox/blob/main/docs/_static/lox.png" width="230">
</div>

# Accelerated logging in JAX

Lox is a lightweight and flexible logging library for JAX that provides a simple interface for logging data during function execution.
Logging is implemented with it's own primitive, which allows it to work seamlessly with JAX's built-in function transformations like `jit` or `vmap`.
The basic logging functionality alone, however, does not provide any benefits over built-in callbacks.
Lox is built around the idea of spooling, which allows you to capture logs generated during function execution and return them alongside the function's output.
While it's obviously possible to implement this functionality yourself, Lox provides a simple and efficient way to do so without having to carry around boilerplate code in your functions.

```python
import jax
import lox

def f(xs):
    lox.log({"xs": xs})
    def scan_step(xs, x):
        xs = xs + x
        lox.log({"xs": carry, "x": x})
        return carry, x
    lox.log({"xs": xs})
    ys, _ = jax.lax.scan(scan_step, 0, xs)
    return ys

xs = jnp.arange(3)
ys, logs = lox.spool(f)(xs)
```
The spooled version of `f` will return both the output of the function and a pytree of all logs generated during execution.
In the example above, `logs` will have the following structure.
```
{
    'x': Array([0, 1, 2], dtype=int32), 
    'xs': Array([[0, 1, 2],
                 [0, 1, 2],
                 [0, 1, 2],
                 [1, 2, 3],
                 [3, 4, 5]], dtype=int32)
}
```

## Features

### General Logging Utility
- **`lox.log`**
  Standard logging functionality. Deault behavior is logging the arguments to the console.
  Leverages JAX's `jax.debug.callback` to enable logging inside jitted functions.

### Function Spooling
- **`lox.spool`**
  Wraps a function so that it returns both its normal output and a pytree of the logs generated during execution.
  Supports various primitivs like `scan`, `cond` or `pjit`·

### Disabling Logging
- **`lox.nolog`**
  Wraps a function such that all logging is disabled entirely.
  This is useful for performance-sensitive code where logging is not needed.

### Support for Logging Frameworks
- **`lox.wandb`**
  Wraps [Weights & Biases](https://wandb.ai/) and makes them compatible with JAX's function transformations.
  Most importantly it supports `vmap` to enable logging of multiple runs in parallel.

- **`lox.neptune`**
  Wraps [Neptune](https://neptune.ai/) and provides similar functionality to the Weights & Biases integration.

## Installation

Lox can be installed via pip directly from the GitHub repository.

```bash
pip install git+https://github.com/huterguier/lox
```

## Quick Start

### General Logging Example

```python
import lox

def f(x):
    lox.log({"x": x})
    def step(carry, x):
        lox.log({"x": x})
        return carry + x, None
    return jax.lax.scan(step, x, jnp.arange(5))

y = f(3) #{"x": 3}
```

### Logging in Jitted Functions

```python
import jax
import lox

@jax.jit
def f(x):
    lox.log({"x": x})
    return x * x

y = f(3) #{"x": 3}
```

### Function Spooling Example

```python
import lox

def f(x, y):
    lox.log({"x": x, "y": y})
    return x * y

z, logs = lox.spool(f)(3, 4)
print("f(x, y)", z) #12
print("Logs:", logs) #{"x": 3, "y": 4}
```
