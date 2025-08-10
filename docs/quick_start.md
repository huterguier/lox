# Quick Start

## What is Lox?

Logging in JAX is notoriously tedious and cumbersome.
JAX is purposefully designed to be a functional programming framework.
As a consequence one is left with 2 main options for logging in Jax.

<style>
    ol > li::marker {
      font-weight: bold;
    }
</style>
<ol>
  <li> Using <a href="https://docs.jax.dev/en/latest/external-callbacks.html">callbacks</a> to log data. While this is the easiest most flexible way to log data, callbacks come with a cost.
  Executing callbacks creates a significant overhead, which can, especially when done frequently, slow down execution tremendously.</li>
  <li> The second option is to treat the logs as a part of the computation graph. While this is the most efficient way to log data, it can be quite tedious to implement, as it
  requires you to manually add the logs as part of the function output. Additionally, this often creates a bloated function signature, which is not ideal for readability and maintainability.</li>
</ol>

Quite often, it is best to use a tradeoff between the two options.
For example you might want to use spool to collect the logs of a single train step and then
use a callback to log the data at the end of the step.


## How does it work?

Lox is not a logging library in the traditional sense.
By default `lox.log` is a no-op, and it is not meant to be used for logging on its own.
The only thing it does is to insert a JAX [primitive](https://docs.jax.dev/en/latest/jax-primitives.html) that specifies that a values that should be logged.
Lox then applies a function transformation that, based on these primitives, modifies the
function to either insert a callback or to collect the logs and return them as part of the function output.


## Example

