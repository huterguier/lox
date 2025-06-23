---
title: 🔪The Sharp Bits🔪
---
# 🔪The Sharp Bits🔪

## Spooling

Spooling is at the heart of Lox but has its own considerations.  
When using `spool`, it is important to understand how the transformation affects the underlying function.

## When to spool vs. when not to

Tracking down the origin of logged data can be difficult when individual values are transformed.  
Hence, spooling should **not** be used when the functionality of the calling function heavily depends on the values returned by `spool`.

A prime example is logging values obtained from evaluation.  
One way to avoid this problem while still supporting the option to spool:

````python
import jax
import lox
import jax.numpy as jnp

def f(xs):
    lox.log({"xs": xs})
    def scan_step(carry, x):
        carry = carry + x
        lox.log({"xs": carry, "x": x})
        return carry, x
    lox.log({"xs": xs})
    ys, _ = jax.lax.scan(scan_step, 0, xs)
    return ys

xs = jnp.arange(3)
ys, logs = lox.spool(f)(xs)
````

Ideally, `spool` is called at the same level where the resulting data is written to disk or passed to the desired logging framework.

The following are two examples illustrating when and when not to use `spool`.

## Conditionals

Whenever you try to log something within a `cond` or conditional block,  
all execution paths **must** produce identical log shapes and structures.

## Loops

Logging inside loops of unknown length can be problematic.

When using `fori_loop`, it depends on whether the loop is reduced to a `scan` or `while_loop`.  
The latter occurs when arguments to either `upper` or `lower` are non-static and can't be inferred during tracing.

In such cases, logging isn't possible and **Lox will raise an error**.
