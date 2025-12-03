import jax

import lox


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


def callback(logs):
    print("Logging:", logs)


mean = 10.0
xs = jax.random.normal(jax.random.key(0), (5,)) + mean
y = lox.tap(f, callback=callback)(xs)

y, logs = lox.spool(f)(xs)
print("Collected Logs:", logs)
