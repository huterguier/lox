<div align="center">
    <img src="https://github.com/huterguier/lox/blob/main/images/lox3.png" width="200">
</div>

# Accelerated logging in JAX

Lox is a lightweight and flexible logging library designed for JAX applications. It provides a simple API for standard logging, debugging within jitted functions, function spooling, and seamless Weights & Biases (wandb) integration.

## Features

### General Logging Utilities
- **`lox.log`**  
  Standard logging functionality. Deault behavior is logging the arguments to the console.
  Leverages JAX's `jax.debug.callback` to enable logging inside jitted functions.

### Function Spooling
- **`lox.spool`**  
  Wraps a function so that it returns both its normal output and a pytree of the logs generated during execution.

### Disabling Logging
- **`lox.nolog`**
  Wraps a function such that all logging is disabled entirely.

## Installation

Lox is not yet on PyPI but you can install it directly from Github.

```bash
pip install git+https://github.com/huterguier/lox
```

## Quick Start

### General Logging Example

```python
import lox

def f(x):
    lox.log({"x": x})
    return x * x

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
