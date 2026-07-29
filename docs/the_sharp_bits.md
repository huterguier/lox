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

If you can't guarantee matching keys across branches, pass `unify=True` to `lox.spool` instead of restructuring your log calls. It fills any key missing from a branch with a NaN/0/False placeholder (matching the shape/dtype of whichever branch does log it) rather than raising. Note that shape/dtype mismatches for a key present in multiple branches always raise, even with `unify=True` — this only papers over *which* keys each branch logs, not conflicting shapes for the same key.

```python
>>> def f(cond, x):
...     def true_fn(x):
...         lox.log({"a": x, "b": x})
...         return x
...     def false_fn(x):
...         lox.log({"a": x})  # no "b" here
...         return x
...     return jax.lax.cond(cond, true_fn, false_fn, x)
>>> y, logs = lox.spool(f, unify=True)(True, 1.0)
```

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

Loggers that aggregate across runs, such as `ConsoleLogger`, tell runs apart by their state. Give
each run its own `init` — in a Python loop or under `vmap` — and each is tracked separately:

```python
# Three runs, whether looped...
for seed in range(3):
    logger_state = logger.init(jax.random.key(seed))
    logger.spool(f, logger_state)(x[seed])

# ...or vmapped. init is vmapped too, so every lane gets its own state.
logger_states = jax.vmap(logger.init)(keys)
jax.vmap(lambda s, xi: logger.spool(f, s)(xi))(logger_states, x)
```

Sharing one state across lanes instead collapses them into a single run, because the lanes end up
fused into one logged array with nothing marking where each begins:

```python
logger_state = logger.init(jax.random.key(0))
logger.spool(jax.vmap(f), logger_state)(x)  # one run, not three
```

## `vmap` Does Not Always Add a Leading Axis

`vmap` only batches values that actually depend on the mapped input. A `lox.log` call whose value
is the same in every lane — a hyperparameter, a schedule value, a metric computed from unbatched
state — is logged **once**, not once per lane:

```python
def f(x):
    lox.log({"batched": x.sum()})     # differs per lane
    lox.log({"constant": jnp.float32(42.0)})  # identical in every lane
    return x

_, logs = lox.spool(jax.vmap(f))(jnp.ones((3, 5)))
logs["batched"].shape   # (3, 1) -- one entry per lane
logs["constant"].shape  # (1,)   -- a single entry for all three lanes
```

This is ordinary JAX batching rather than a lox behavior, but it is easy to trip over when sweeping
seeds and expecting `n` copies of everything.

More generally, the leading axis of a logged value is not a "time" axis. It fuses scan iterations,
`vmap` lanes and separate `lox.log` call sites together, and how many leading axes there are varies
— a scan adds one, `vmap` can add another. Indexing it positionally (`logs["loss"][-1]` as "the
final value") is therefore unreliable; prefer order-independent reductions such as `.reduce("mean")`
unless you know exactly which transformations produced the array.

## Selective Logging

`lox.tap`, `lox.spool`, `lox.strip`, and `lox.keep` all accept `argnames` and `tags` to filter
which `lox.log` calls they act on. This is useful when you have many `lox.log` calls but only care
about a subset of them for a specific task.

- **`argnames`**: specific keys from your log dictionaries.
- **`tags`**: strict filtering based on tags provided in `lox.log`.

For each of `argnames`/`tags`, `None` means "no restriction on this axis" — the default, meaning
`tap`/`spool` observe everything and `strip`/`keep` act on their full default (`strip` removes
everything, `keep` removes nothing). Passing an iterable — **even an empty one** — switches that
axis to "restrict to exactly this," so `argnames=[]` or `tags=[]` match nothing: `tap`/`spool`
observe nothing, `strip` removes nothing, `keep` keeps nothing. When both `argnames` and `tags`
are given together, a log entry must match **both** to be selected (logical AND), not either.

```python
lox.log({"a": x, "b": x}, tags=["train"])
lox.log({"c": x}, tags=["eval"])

# argnames=["a"] and tags=["train"] combine with AND: only "a" matches both.
lox.strip(f, argnames=["a"], tags=["train"])   # removes only "a", keeps "b" and "c"
lox.keep(f, argnames=["a"], tags=["train"])    # keeps only "a", removes "b" and "c"
```

```python
# In your code
lox.log({"loss": loss}, tags=["metric"])
lox.log({"gradient_norm": grad_norm}, tags=["debug"])

# Only tap into metrics
lox.tap(f, tags=["metric"])(x)
```
