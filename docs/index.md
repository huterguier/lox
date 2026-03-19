# Logging in JAX

[`lox`](https://github.com/huterguier/lox) is a lightweight and flexible logging library for [JAX](https://github.com/jax-ml/jax).
All you need to do is decorate your code with `lox.log` statements and `lox` does the rest.
Using JAX's intermediate function representation Lox can dynamically insert callbacks to log you data, or collect the logs that would have been generated during the execution and return them as part of the output of your function.
While it's obviously possible to implement this functionality yourself, `lox` provides a simple and efficient way to do so without having to carry around boilerplate code in your functions.

## Features
<div class="feature-box">

🔌 **Plug-and-Play:**  Simply add `lox.log` statements where you need them. `lox` handles all the complex boilerplate of plumbing data through JAX's transformations, keeping your function signatures clean and focused on the logic.

</div>

<div class="feature-box">

📦 **Automatic Extraction:**  Instead of explicitly returning data from you functions, `lox.spool` automatically "spools up" all logs generated during a function's execution. It collects them and returns them as a single `logdict` alongside the function's original output.

</div>

<div class="feature-box">

📡 **Dynamic Callbacks:**  Using `lox.tap`, you can "tap into" a JAX-transformed function using custom callbacks. This is ideal for live monitoring and debugging without halting execution.

</div>

<div class="feature-box">

✅ **`vmap` over Seeds:**  Built on its own JAX primitive, `lox` works effortlessly with core transformations like `jit`, `scan`, and `vmap`.

</div>

<div class="feature-box">

📊 **Experiment Loggers:**  Includes built-in loggers that seamlessly pipe your metrics to popular experiment tracking platforms including [`wandb`](https://wandb.ai/) and [`neptune`](https://neptune.ai/), which are also fully compatible with `vmap`.

</div>


## Installation
`lox` can be installed directly from this GitHub repository.
```bash
pip install git+https://github.com/huterguier/lox
```
By default `lox` comes without any of the external experiment loggers. Make sure to include the optional dependencies or to install them manually.
```bash
pip install "lox[wandb,neptune] @ git+https://github.com/huterguier/lox"
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Introduction

quick_start
why_lox
the_sharp_bits
examples
contributing
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: API

design_decisions
api
```
