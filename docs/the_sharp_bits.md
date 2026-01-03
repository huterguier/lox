---
title: 🔪The Sharp Bits🔪
---
# 🔪The Sharp Bits🔪

This page covers edge cases, potential pitfalls, and advanced usage of `lox`.
While `lox` aims to be as seamless as possible, there are some important details to be aware of when using it in complex scenarios.
Overall, understanding these nuances will help you avoid common mistakes and make the most of `lox`'s capabilities.


## Conditionals

Whenever you use `lox.log` within a `jax.lax.cond` or `if` statement (which gets traced to `cond`), all execution paths **must** produce identical log shapes and structures.
This is because JAX requires static output shapes for compiled functions.

If one branch logs `{"a": 1}` and the other logs `{}`, `lox` (and JAX) will raise an error because the return structure of the `cond` primitive would be inconsistent.

## Loops

Logging inside loops behaves differently depending on the loop primitive used.

- **`jax.lax.scan`**: Supported by both `spool` and `tap`. Since the number of iterations is known, `spool` can pre-allocate memory for the logs.
- **`jax.lax.while_loop`**: Supported by `tap`, but **not** by `spool`. 
  - `tap` works because it executes a callback at runtime for each iteration.
  - `spool` fails (or warns and returns empty logs) because the number of iterations is not known at compile time, so JAX cannot determine the shape of the resulting log array.
  
**Note on `fori_loop`**: JAX's `fori_loop` is sometimes lowered to `scan` and sometimes to `while_loop`. 
If the lower and upper bounds are static, it acts like `scan` and `spool` works. 
If they are dynamic, it acts like `while_loop` and `spool` will not work.

## `tap` vs `spool` Performance

- **`tap`**: Uses host-callbacks. Great for debugging and printing to stdout. However, frequent callbacks (e.g., inside a tight loop on GPU) can severely degrade performance by forcing synchronization between device and host. Use sparingly in performance-critical code.
- **`spool`**: Keeps data on the device. It modifies the function to return logs as extra outputs. This is generally much faster than `tap` for collecting data, but it consumes device memory.

## JIT Compilation

`lox` transformations modify the `jaxpr` of the function.
Since the overall e

- If you `jit` a function *after* applying `spool`, the logs become part of the compiled output.
- If you `spool` a function that has already been `jit`-ted, it will use the existing compiled version without retriggering compilation.
```python
@jax.jit
def f(x):
    lox.log({"x": x})
    return x

x = 1.0
y = f(x) # this will trigger jit compilation of f
y = f(x) # this will use the compiled version
y, logs = lox.spool(f)(x) # spool retriggers compilation
```


## Loggers and State

When using loggers (like `SaveLogger` or `WandbLogger`), remember that they are stateful. 
You must initialize them and pass the state to the transformation.

```python
logger = lox.loggers.SaveLogger("/tmp/logs")
logger_state = logger.init(jax.random.key(0))
y = logger.spool(f, logger_state)(inputs)
```
The `spool` method on a logger typically collects all logs first and then writes them (e.g., to disk) in one go after the function returns, whereas `tap` might write them incrementally.

## Selective Logging

Both `lox.tap` and `lox.spool` allow you to filter what gets logged using `argnames` and `tags`.
This is useful when you have many `lox.log` calls but only care about a subset of them for a specific task.

- **`argnames`**: specific keys from your log dictionaries.
- **`tags`**: strict filtering based on tags provided in `lox.log`.

```python
# In your code
lox.log({"loss": loss}, tags=["metric"])
lox.log({"gradient_norm": grad_norm}, tags=["debug"])

# Only tap into metrics
lox.tap(f, tags=["metric"])(x)
```
