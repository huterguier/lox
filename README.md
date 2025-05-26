<div align="center">
    <img src="https://github.com/huterguier/lox/blob/dev/images/lox3.png" width="200">
</div>

# Accelerated logging in JAX

LOX is a lightweight and flexible logging library designed for JAX applications. It provides a simple API for standard logging, debugging within jitted functions, function spooling, and seamless Weights & Biases (wandb) integration.

## Features

### General Logging Utilities
- **`lox.log`**  
  Standard logging functionality for non-jitted contexts. Use it to print messages, debug data, or track custom events.

### Logging in Jitted Functions
- Leverages JAX's debugging tools (`jax.debug.print` and `jax.debug.callback`) to enable logging inside jitted functions.  
  **Note:** This method may introduce performance overhead.

### Function Spooling
- **`lox.spool`**  
  Wraps a function so that it returns both its normal output and a pytree of the logs generated during execution.

### Weights & Biases (wandb) Integration
- **`lox.wandb.log`**  
  Integrates directly with wandb, enabling logging within jitted functions and across JAX primitives to facilitate streamlined experiment tracking.

### Automatic Wandb Logging with Spooling
- **`jax.wandb.spool`**  
  Automatically wraps a function to log computed values to wandb after execution, combining the benefits of function spooling with seamless wandb integration.

## Installation

Install LOX via pip:

```bash
pip install lox
```

## Quick Start

### General Logging Example

```python
import lox

# Standard logging
lox.log("Starting computation...")

def compute(x):
    lox.log("Computing square for {}", x)
    return x * x

result = compute(3)
```

### Logging in Jitted Functions

```python
import jax
import lox

@jax.jit
def jitted_compute(x):
    # Logging via JAX debugging tools
    lox.log("Computing square of {}", x)
    return x * x

result = jitted_compute(3)
```

### Function Spooling Example

```python
import lox

def multiply_and_log(x, y):
    lox.log("Multiplying {} and {}", x, y)
    return x * y

# Wrap the function to capture logs
result, logs = lox.spool(multiply_and_log)(3, 4)
print("Result:", result)
print("Logs:", logs)
```

### Wandb Integration Example

```python
import lox

# Log metrics to wandb
lox.wandb.log({"loss": 0.05, "accuracy": 0.98})
```

### Automatic Wandb Logging with Spooling

```python
import jax
import lox

def compute_metrics(x):
    result = x * x
    lox.log("Computed metric: {}", result)
    return result

# Wrap the function to automatically log to wandb
result = jax.wandb.spool(compute_metrics)(5)
```

## License

LOX is released under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

For support or inquiries, please open an issue on the repository or contact the maintainers directly.

--- 

This README outlines the core functionality of LOX and should help you quickly integrate logging into your JAX workflows.
