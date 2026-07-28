import jax
import jax.numpy as jnp
import pytest

from lox.io import load, save


@pytest.fixture(
    params=[
        {"a": jnp.array([1, 2, 3]), "b": jnp.array([[1.0, 2.0], [3.0, 4.0]])},
        {"x": jnp.zeros((5, 5)), "y": jnp.ones((2, 3, 4))},
        {"nested": {"c": jnp.array([10, 20, 30])}},
        {"root/nested": jnp.array([42])},
    ]
)
def logs(request):
    return request.param


def test_save_load(tmp_path, logs):
    path = str(tmp_path / "test_logs.pkl")
    save(logs, path)
    loaded_logs = load(path)

    for key in logs:
        if isinstance(logs[key], dict):
            for subkey in logs[key]:
                assert jnp.array_equal(logs[key][subkey], loaded_logs[key][subkey]), (
                    f"Mismatch in nested key: {key}->{subkey}"
                )
        else:
            assert jnp.array_equal(logs[key], loaded_logs[key]), (
                f"Mismatch in key: {key}"
            )


def test_vmap_save_load(tmp_path, logs):
    path = str(tmp_path / "vmap_logs.pkl")
    keys = jnp.stack([jax.random.key(i) for i in range(3)])
    jax.vmap(lambda key: save(logs, path, mode="w", key=key))(keys)
    loaded_logs = load(path, keys)

    loaded_logs = jax.tree.map(lambda x: x[0], loaded_logs)
    for key in logs:
        if isinstance(logs[key], dict):
            for subkey in logs[key]:
                assert jnp.array_equal(logs[key][subkey], loaded_logs[key][subkey]), (
                    f"Mismatch in nested key: {key}->{subkey}"
                )
        else:
            assert jnp.array_equal(logs[key], loaded_logs[key]), (
                f"Mismatch in key: {key}"
            )


def test_save_invalid_path(logs):
    with pytest.raises(Exception):
        save(logs, "/invalid_path/test_logs.pkl")


def test_save_empty_logs(tmp_path):
    empty_logs = {}
    path = str(tmp_path / "empty_logs.pkl")
    save(empty_logs, path)
    loaded_logs = load(path)
    assert loaded_logs == empty_logs, "Loaded logs should be empty dictionary"


def test_load_missing_path_raises(tmp_path):
    path = str(tmp_path / "never_saved")
    with pytest.raises(FileNotFoundError):
        load(path)


def test_load_argnames_list(tmp_path):
    path = str(tmp_path / "logs")
    data = {"carry": jnp.ones(3), "x": jnp.zeros(3)}
    save(data, path)
    loaded_logs = load(path, argnames=["carry"])
    assert set(loaded_logs.keys()) == {"carry"}
    assert jnp.array_equal(loaded_logs["carry"], data["carry"])


def test_load_argnames_bare_string_is_exact_match(tmp_path):
    path = str(tmp_path / "logs")
    data = {"carry": jnp.ones(3), "c": jnp.zeros(3)}
    save(data, path)
    loaded_logs = load(path, argnames="carry")
    assert set(loaded_logs.keys()) == {"carry"}
    assert jnp.array_equal(loaded_logs["carry"], data["carry"])
