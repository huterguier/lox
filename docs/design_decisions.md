---
title: Design Decisions
---

# Design Decisions

`lox` is designed to provide a seamless logging experience in JAX, adhering to its functional paradigm while minimizing boilerplate. This document outlines the key architectural choices and design philosophies behind the library.

## Motivation
Logging in JAX is notoriously difficult due to its pure functional nature and the way transformations like `jit`, `vmap`, and `scan` work. Common approaches often involve:
1. **Explicit Return:** Manually plumbing metrics through every function return. This clutters function signatures and logic.
2. **Side-Effects (`host_callback` or `debug.callback`):** Using callbacks directly. While useful for monitoring, it can be difficult to collect and process these logs in a structured way (e.g., for reduction or saving).

`lox` aims to provide the best of both worlds: the simplicity of "add a log statement anywhere" with the power of structured collection and real-time monitoring.

## Core Philosophy: Jaxpr Manipulation
The central design decision of `lox` is to treat logging as a **program transformation** rather than a side-effecting operation.
Instead of relying on global state or immediate execution of callbacks, `lox.log` inserts a custom JAX primitive (`lox_p`) into the function's intermediate representation (jaxpr).

By deferring the interpretation of these log points, `lox` can decide how to handle them based on the desired transformation:
- **`lox.tap`**: Re-writes the jaxpr to insert `jax.debug.callback` at each `lox_p` site.
- **`lox.spool`**: Re-writes the jaxpr to collect the logged values and return them as additional function outputs.
- **`lox.strip`**: Removes all `lox_p` primitives from the jaxpr, effectively neutralizing logging with zero runtime overhead.
- **`lox.keep`**: The whitelist complement of `lox.strip` — removes every `lox_p` primitive *except* the ones matching `argnames`/`tags`, instead of matching ones. With no filter it is a no-op, mirroring `strip`'s no-filter default of removing everything.

## The `lox_p` Primitive
The `lox` primitive is designed to be as transparent as possible:
- **Functional Purity:** By default, it returns its input values, maintaining the functional purity of the user's code.
- **Effectful Abstract Evaluation:** It is registered with `DebugEffect`, which signals to JAX that this primitive might have side effects when transformed (important for `tap`).
- **Transformation Support:** It implements batching (`vmap`) and JVP rules, ensuring it works seamlessly with JAX's core transformations.

## Dual API: Tap vs. Spool
`lox` provides two primary ways to consume logs, catering to different use cases.

### `lox.tap` (Real-time Monitoring)
`tap` is designed for live inspection. It uses `jax.debug.callback` to send data back to the Python host as it is encountered during execution. This is ideal for:
- Console logging of progress.
- Live plotting/monitoring.
- Debugging values inside `jit` or `scan`.

### `lox.spool` (Structured Collection)
`spool` is designed for efficiency and batch processing. It transforms the function so that it returns a `logdict` containing all logged values.
- **Efficiency:** Within `scan`, `spool` automatically handles the stacking of logs across iterations, returning them as a single array.
- **Post-processing:** Collected logs can be sliced, reduced (e.g., `jnp.mean`), or saved after the function execution.
- **Static Shape Requirement:** Since `spool` modifies the output signature of a JAX function, it requires that the number of log events is statically known. This is why `spool` does not support `while_loop` with dynamic termination.

## Data Structure: `logdict`
Logs in `lox` are not just raw values; they are encapsulated in a `logdict`.
- **Pytree Compatibility:** `logdict` is registered as a JAX pytree, allowing it to be passed in and out of JAX-transformed functions.
- **Tag-Based Filtering:** Individual `lox.log` calls can be tagged with arbitrary strings. `tap`, `spool`, `strip`, and `keep` can all filter on these tags, so a single set of log statements can serve multiple downstream consumers without duplicating log calls. `argnames`/`tags` combine with logical AND when both are given; `None` means no restriction on that axis, while an empty list restricts to exactly nothing.
- **Rich Interface:** `logdict` provides a dictionary-like interface but adds methods for `slice`, `reduce`, and merging (`+`), facilitating easy post-processing.

## Logger Abstraction
`lox` separates the *instrumentation* (`lox.log`) from the *output backend*.
- The `Logger` base class defines a standard interface for backend integrations (e.g., `ConsoleLogger`, `WandBLogger`, `SaveLogger`).
- Loggers can be easily swapped or combined using `MultiLogger` without changing the underlying function logic.
- **State Management:** Loggers use a `LoggerState` pattern, consistent with JAX's state-handling conventions (e.g., for `init` and `log` calls).

## Support for JAX Transformations
A non-negotiable goal for `lox` is full support for JAX's core transformations.
- **`vmap`**: `lox` handles batching by automatically expanding the dimensions of logged data.
- **`scan` / `fori_loop`**: `lox` handles the sequential nature of these loops, either by streaming (in `tap`) or stacking (in `spool`).
- **`jit`**: `lox` transformations work at the jaxpr level before compilation, ensuring that logging logic is baked into the compiled artifact.