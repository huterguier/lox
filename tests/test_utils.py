import jax
import jax.numpy as jnp

from lox.utils import get_path


def test_get_path_simple_seed_is_readable():
    for n in range(1, 6):
        assert get_path("/tmp/x", jax.random.key(n)) == f"/tmp/x/{n}"


def test_get_path_no_collision_across_key_words():
    kd1 = jnp.array([1, 23], dtype=jnp.uint32)
    kd2 = jnp.array([12, 3], dtype=jnp.uint32)
    k1 = jax.random.wrap_key_data(kd1)
    k2 = jax.random.wrap_key_data(kd2)
    assert get_path("/tmp/x", k1) != get_path("/tmp/x", k2)


def test_get_path_deterministic():
    key = jax.random.key(0)
    assert get_path("/tmp/x", key) == get_path("/tmp/x", key)
