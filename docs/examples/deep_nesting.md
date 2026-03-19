# Deep Nesting

`lox` is particularly useful when dealing with deeply nested function calls.
Here's an example that demonstrates how `lox` can help manage complexity in such scenarios.
Let's consider the following set of functions that call each other in a nested manner and assume that we are interested in the intermediate variable `a` in function `x`.

```python
def f(x):
    a = x * 2
    y = a + 1
    return y

def g(x):
    y = f(x) * 2
    return y

def h(x):
    y = g(x) - 3
    return y

y = h(x)
```

Without `lox`, tracking the value of `a` through these nested calls can be challenging.
We need to modify each function to return `a` along with `y`, which can clutter the code.
But not only that, we also need to adjust all subsequent function calls to handle the additional return value.
This can lead to significant code bloat and make the code harder to read and maintain.

```python
def f(x):
    a = x * 2
    y = a + 1
    return y, a

def g(x):
    y_f, a = f(x)
    y = y_f * 2
    return y, a

def h(x):
    y_g, a = g(x)
    y = y_g - 3
    return y, a

y, a = h(x)
```

With `lox`, we can simplify this process significantly.
By using `lox`, we can capture the value of `a` in `f` without modifying the function signatures or return values.
Note that `spool` always adds a leading dimension to the logs in order to account for multiple values with the same name being logged during a single call.
That is why we access `logs["a"][0]` to get the first logged value of `a`.

```python
def f(x):
    a = x * 2
    lox.log({"a": a})
    y = a + 1
    return y

def g(x):
    y = f(x) * 2
    return y

def h(x):
    y = g(x) - 3
    return y

y, logs = lox.spool(h)(x)
a = logs["a"][0]
```
