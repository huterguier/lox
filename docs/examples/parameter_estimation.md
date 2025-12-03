# Parameter Estimation

In the following example, we will illustrate how to use Lox to log data in a JAX function.
We will first define a simple pure JAX function,
  then we will decorate it with `lox.log` statements to specify which values we want to log,
  and finally we will use Lox's function transformations to access the logs.

### 1. A simple JAX function

The example is a simple function that computes the mean and standard deviation of a list of numbers using gradient descent.
The function `f` takes a list of numbers `xs` and returns the mean and standard deviation of the numbers.
It does so by iterating over the individual values `x` in `xs` and updating the mean and standard deviation using gradient descent.

```python
>>> import jax
>>> import jax.numpy as jnp

>>> def f(xs):
...     mean, std = 0.0, 1.0
...     params = (mean, std)
...
...     def step(params, x):
...         def loss(params):
...             mean, std = params
...             loss_mean = (mean - x) ** 2
...             std_x = jnp.abs(mean - x) / jnp.sqrt(2.0 / jnp.pi)
...             loss_std = (std_x - std) ** 2
...             return loss_mean + loss_std
...
...         gradient = jax.grad(loss)(params)
...         params = jax.tree_util.tree_map(lambda p, g: p - 1e-4 * g, params, gradient)
...         return params, None
...
...     (mean, std), _ = jax.lax.scan(step, params, xs)
...     return mean, std
```


### 2. Decorating the function with `lox.log`

In order to use Lox, we need to decorate the function with `lox.log` statements to specify which values we want to log.
`lox.log` takes a single positional argument, which is the dictionary of values to log.
All additional keyword arguments are treated as timesteps and will be logged as well.
For the sake of simplicity, we wont use any timesteps in this example, but you can refer to the [API documentation](api.md) for more details on how to use timesteps.
In this example, we are interested in logging the mean and standard deviation of the parameters, as well as the loss values for each step.
-- we only use scalars in this example but can be pytrees but make sure same shape
While we only use scalars in this example, Lox can also log Arrays and PyTrees.
When logging with the same key multiple times, it is important to ensure that the values have the same shape as `lox.spool` will concatenate the values along the first axis.

```python
>>> import lox

>>> def f(xs):
...     mean, std = 0.0, 1.0
...     params = (mean, std)
...
...     def step(params, x):
...         def loss(params):
...             mean, std = params
...             loss_mean = (mean - x) ** 2
...             std_x = jnp.abs(mean - x) / jnp.sqrt(2.0 / jnp.pi)
...             loss_std = (std_x - std) ** 2
...             lox.log({
...               "loss_mean": loss_mean, 
...               "loss_std": loss_std, 
...               "loss": loss_mean + loss_std
...             })
...             return loss_mean + loss_std
...
...         gradient = jax.grad(loss)(params)
...         params = jax.tree_util.tree_map(lambda p, g: p - 1e-4 * g, params, gradient)
...         lox.log({"mean": params[0], "std": params[1]})
...         return params, None
...
...     (mean, std), _ = jax.lax.scan(step, params, xs)
...     return mean, std
```

### 3. Accessing the logs using `lox.tap`

Now that we have decorated the function with `lox.log`, we can use Lox's function transformations to access the data.
The transformation `lox.tap` lets you "tap into" function execution by attaching a callback that receives logs as they're generated. It streams logs in real time, making it great for debugging or live monitoring.

```python
>>> def callback(logs):
...     print("Logging:", logs)
>>> y = lox.tap(f, callback=callback)(xs)

Logging:  {'loss': 460.6522216796875, 'loss_mean': 192.3193817138672, 'loss_std': 268.3328552246094}
Logging:  {'mean': 0.0068796598352491856, 'std': 1.0032762289047241}
Logging:  {'loss': 546.866943359375, 'loss_mean': 227.07217407226562, 'loss_std': 319.7947692871094}
Logging:  {'mean': 0.014375997707247734, 'std': 1.0068527460098267}
Logging:  {'loss': 133.441650390625, 'loss_mean': 59.05677795410156, 'loss_std': 74.3848648071289}
Logging:  {'mean': 0.018074849620461464, 'std': 1.0085777044296265}
```

The great thing about `lox.tap` is that you can selectively log only the values you are interested in, without cluttering your function with print statements or other logging code.
By setting the keyword argument `argnames`to a desired sequence of strings, you can specify which values to log.
If, for example, we are only interested in the mean and standard deviation, we can do the following:

```python
>>> y = lox.tap(f, callback=callback, argnames=["mean", "std"])(xs)
Logging:  {'mean': 0.0068796598352491856, 'std': 1.0032762289047241}
Logging:  {'mean': 0.014375997707247734, 'std': 1.0068527460098267}
Logging:  {'mean': 0.018074849620461464, 'std': 1.0085777044296265}
```

### 4. Collecting logs using `lox.spool`

The other fundamental transformation `lox.spool`, "spools up" all logs during execution and returns them alongside the function's output. 
This is especially useful when frequent callbacks would be too expensive. 

```python
>>> y, logs = lox.spool(f)(xs)
>>> print("Collected Logs:", logs)
Collected Logs: {
    'mean': array([0.00687966, 0.014376  , 0.01807485]),
    'std': array([1.0032762 , 1.0068527 , 1.0085777 ]),
    'loss_mean': array([192.31938, 227.07217,  59.05678]),
    'loss_std': array([268.33286, 319.79477,  74.384865]),
    'loss': array([460.65222, 546.86694, 133.44165])
}
```

Often times it can be useful to use a combination of `lox.tap` and `lox.spool`.

## 5. Using Loggers

Lox provides a variety of loggers that can be used to write the logs to different backends.
A logger can also be passed directly to `lox.spool`.
Doing this will cause the logs to be written to the logger instead of being returned as part of the function output.
