<div align="center">
    <img src="https://github.com/huterguier/lox/blob/main/images/lox3.png" width="200">
</div>

# Accelerated logging in JAX

Lox is a lightweight and flexible logging library for JAX that provides a simple interface for logging data during function execution.
Logging is implemented with it's own primitive, which allows it to work seamlessly with JAX's built-in function transformations like `jit` or `vmap`.
The basic logging functionality alone, however, does not provide any benefits over built-in callbacks.
Lox is built around the idea of spooling, which allows you to capture logs generated during function execution and return them alongside the function's output.
While it's obviously possible to implement this functionality yourself, Lox provides a simple and efficient way to do so without having to carry around boilerplate code in your functions.

## Features

### General Logging Utilities
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
  Integrates with [Weights & Biases](https://wandb.ai/) to log data in a structured way.
  Automatically handles logging of function arguments and outputs, as well as custom logs and most importantly supports `vmap` to enable logging of multiple runs in parallel.

- **`lox.neptune`**  
  Integrates with [Neptune](https://neptune.ai/) to log data in a structured way.
  Similar to the Weights & Biases integration, it automatically handles logging of function arguments and outputs.

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
