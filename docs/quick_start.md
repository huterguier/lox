# Quick Start

Logging in JAX can be incredibly tedious and cumbersome.
JAX is purposefully designed to be fully functional,
  and as a consequence one is left with 2 main options for logging in Jax.

<style>
    ol > li::marker {
      font-weight: bold;
    }
</style>
<ol>
  <li> 
    Using <a href="https://docs.jax.dev/en/latest/external-callbacks.html">callbacks</a> to log data. 
    While this is the easiest most flexible way to log data, callbacks come with a cost.
    Executing callbacks creates a non-negligable overhead, which can, especially when done frequently, 
      slow down execution tremendously.
    Moreover, these callbacks need to be inserted manually, which can clutter the code and make it less readable.
  </li>
  <li> 
    The second option is to treat the logs as a part of the computation graph. 
    While this is the most efficient way to log data, it can be quite tedious to implement, as it
      requires you to manually add the logs as part of the function output. 
    Additionally, this usually creates a bloated function signature, 
      which is not ideal for readability and maintainability.
  </li>
</ol>


## What is `lox`?

`lox` is a lightweight logging library for JAX that aims to dramatically simplify these two approaches.
It takes care of all the boilerplate code that is usually required.
With `lox`, you can easily log data in a JAX function without cluttering your code with print statements or callbacks.
`lox` provides two fundamental function transformations, `lox.tap` and `lox.spool`, that
  allow you to either stream logs in real time using a callback or collect all logs and return them as part of the function output.
Lox also provides a variety of loggers that can be used to write the logs to different backends.

## How does it work?

`lox` is not a logging library in the traditional sense.
By default the core function `lox.log` is a no-op, and it is not meant to be used for logging on its own.
The only thing it does is to insert a JAX [primitive](https://docs.jax.dev/en/latest/jax-primitives.html),
  that specifies that the values that you want to log in a dictionary format.
Lox then applies a function transformation that, based on these primitives, modifies the
    function to either insert a callback or to collect the logs and return them as part of the function output.


## Example

In the following example, we will illustrate how to use Lox to log data in a JAX function.
We will first define a simple pure JAX function,
  then we will decorate it with `lox.log` statements to specify which values we want to log,
  and finally we will use Lox's function transformations to access the logs.
To illustrate how Lox works, 
  we will define a simple JAX function that performs a few optimization steps using gradient descent.
The function takes in a sequence of data points and approximates their mean by minimizing the mean squared error.

```python
import jax
import lox

def f(xs):
    def step(mean, x):
        def loss(mean):
            diff = mean - x
            loss = (diff) ** 2
            return loss
        gradient = jax.grad(loss)(mean)
        params = jax.tree_util.tree_map(lambda p, g: p - 1e-2 * g, mean, gradient)
        return params, None
    mean = 0.0
    mean, _ = jax.lax.scan(step, mean, xs)
    return mean
```


### 1. Decorating the function with `lox.log`

In order to use Lox, we need to decorate the function with `lox.log` statements. 
  These specify which values we want to log during the function execution.
`lox.log` takes a single positional argument, which is the dictionary of values to log.
All additional keyword arguments are treated as timesteps and will be logged as well.
For the sake of simplicity, we wont use any timesteps in this example,
  but you can refer to the [API documentation](api.md) for more details on how to use timesteps.
In this example, 
  we are interested in logging the signed difference between the current mean and the data point.

```{code-block} python
:emphasize-lines: 6
def f(xs):
    def step(mean, x):
        def loss(mean):
            diff = mean - x
            loss = (diff) ** 2
            lox.log({"diff": diff})
            return loss
        gradient = jax.grad(loss)(mean)
        params = jax.tree_util.tree_map(lambda p, g: p - 1e-2 * g, mean, gradient)
        return params, None
    mean = 0.0
    mean, _ = jax.lax.scan(step, mean, xs)
    return mean
```


### 2. Collecting logs using `lox.spool`

Now that we have decorated the function with `lox.log`, 
  we can use function transformations to access the data.
`lox.spool` is a function transformation "spools up" all logs during execution and returns them alongside the function's output. 
This is especially useful when frequent callbacks would be too expensive. 
The collected logs can then be handled after the function execution.

```python
>>> mean = 10.0
>>> xs = jax.random.normal(jax.random.key(0), (3,)) + mean
>>> y, logs = lox.spool(f)(xs)
>>> print("Collected Logs:", logs)
Collected Logs: {'diff': Array([-11.6226425, -11.792812, -9.098096, -9.2711115, -9.340398], dtype=float32)}
```

In this simple example collecting the logs manually would not be too difficult.
However, in more complex scenarios with nested functions and multiple logging points,
  manually collecting logs can become quite tedious and error-prone.
`lox.spool` takes care of all the boilerplate code for you,

### 3. Accessing the logs using `lox.tap`

The second transformation `lox.tap` let's you "tap into" function execution by attaching a callback that receives logs as they're generated. 
It streams logs in real time, making it great for debugging or live monitoring.
The cool thing bout it is that you can define the callback function once, 
  and `lox` automatically inserts it at every logging point in the function.

```python
>>> def callback(logs):
...     print("Logging:", logs, flush=True)
>>> y = lox.tap(f, callback=callback)(xs)

Logging: {'diff': Array([-11.6226425], dtype=float32)}
Logging: {'diff': Array([-11.792812], dtype=float32)}
Logging: {'diff': Array([-9.098096], dtype=float32)}
Logging: {'diff': Array([-9.2711115], dtype=float32)}
Logging: {'diff': Array([-9.340398], dtype=float32)}
```

Another great thing about `lox.tap` is that you can also selectively log only the values you are interested in.
By setting the keyword argument `argnames` to a desired iterable of strings, 
  you can specify which values to log.
The selection will be done during compiliation time, 
  so there is no runtime overhead for filtering out unwanted logs.

### 4. Using Loggers

`lox` provides a variety of loggers that can be used to write the logs to different backends.
Loggers also support the two main function transformations, `lox.tap` and `lox.spool`.
For example, you can use the `lox.loggers.SaveLogger` to save the logs to a file.
```python
from lox.loggers import SaveLogger
logger = SaveLogger("logs.pkl")
logger_state = logger.init(jax.random.key(0))
y = logger.tap(f, logger_state)(xs)
```
These loggers are also fully compatible with `vmap`.
In the following example, 
  we will use the `WandBLogger` to log the data of 5 parallel runs to Weights and Biases.
```python
from lox.wandb import WandBLogger
logger = WandBLogger(project="lox", name="experiment")
def g(key):
    xs = jax.random.normal(key, (10,)) + mean
    logger_state = logger.init(key)
    y = logger.tap(f, logger_state)(xs)
    return xs
keys = jax.random.split(jax.random.key(0), 5)
y = jax.vmap(g)(keys)
```
