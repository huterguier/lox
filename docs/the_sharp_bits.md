---
title: 🔪The Sharp Bits🔪
---
# 🔪The Sharp Bits🔪

## Selective Logging



## When not to use `spool`

Tracking down the origin of logged data can be difficult when individual values are transformed.
Hence, spooling should **not** be used when the functionality of the calling function heavily depends on the values returned by `spool`.

A prime example is logging values obtained from evaluation.  

## How to `vmap` over strings?
As you probably know, JAX does not support strings. 
However a lot of times it can be useful to vmap over strings, for example when running different seeds in parallel and assigning a different path or name to each run.
Lox provides a custom string wrapper that encodes strings as JAX arrays, allowing you to use them with `vmap`.
````python
import lox
names = jax.vmap(lambda k: lox.StringArray(f"run_{k}"))(jax.numpy.arange(10))
````


## Conditionals

Whenever you try to log something within a `cond` or conditional block,  
all execution paths *must* produce identical log shapes and structures.
If this is not the case, `lox` will raise an error.

## Loops

Logging inside loops of unknown length can be problematic.

When using `fori_loop`, it depends on whether the loop is reduced to a `scan` or `while_loop`.  
The latter occurs when arguments to either `upper` or `lower` are non-static and can't be inferred during tracing.
In such cases, logging isn't possible and **Lox will raise an error**.

## `vmap` of `spool`

When using `vmap` over functions that contain `spool` calls,
Lox will automatically batch the logged values along a new leading axis.
This means that if you `vmap` over a function that logs a scalar value,
the resulting logged value will be an array with shape `(N,)`, where `N` is the size of the `vmap` batch.
