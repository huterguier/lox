
# Lox <small><em>(Logs)</em></small>

Lox is a loggin library for [JAX](https://docs.jax.dev/en/latest/) that provides a simple and flexible way to log metrics, and other data during the execution of JAX programs. It is designed to be easy to use and integrate with existing JAX workflows, while also providing powerful features for advanced logging needs.

## What is Lox?

Logging in JAX is notoriously tedious and cumbersome.
JAX is purposefully designed to be a functional programming framework, which means that it does not have built-in support for side effects, such as logging.
As a consequence one is left with 2 main options for logging in Jax.
1. Using callbacks to log data. While this is the easiest most flexible way to log data, callbacks come with a cost. 
Executing callbacks creates a significant overhead, which can, especially when done frequently, slow down execution tremendously.
2. The second option is to treat the logs as a part of the computation graph. While this is the most efficient way to log data, it can be quite tedious to implement, as it
requires you to manually add the logs as part of the function output. 


Lox is not a logging library in the traditional sense.

By default `lox.log` is a no-op, and it is not meant to be used for logging on its own.
You might be wondering

That is mainly because loggin in JAX can be quite expensive, especially when done frequently.

At its core, 3 functions: `lox.log`, `lox.spool` and `lox.tap`





## Features

- **Simple API**: Lox provides a simple and intuitive API for logging metrics and other data during the execution of JAX programs.
- **Flexible**: Lox is designed to be flexible and can be easily integrated into existing JAX workflows.
- **Advanced Features**: Lox provides advanced features such as logging to multiple backends, custom loggers, and more.
- **JAX Compatible**: Lox is fully compatible with JAX and can be used with any JAX program.


## Installation

```bash
pip install jax-lox
# Alternatively, you can install the latest version from GitHub.
pip install git+https://github.com/huterguier/lox
```

Add your content using **Markdown** syntax via [MyST](https://myst-parser.readthedocs.io/).  
See the [MyST docs](https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html) for supported syntax.

```{toctree}
:hidden:
:maxdepth: 2

quick_start
the_sharp_bits
api
```
