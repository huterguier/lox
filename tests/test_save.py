import jax.numpy as jnp
import pytest

from lox.save import load, save


@pytest.fixture(
    params=[
        {"a": jnp.array([1, 2, 3]), "b": jnp.array([[1.0, 2.0], [3.0, 4.0]])},
        {"x": jnp.zeros((5, 5)), "y": jnp.ones((2, 3, 4))},
        {"nested": {"c": jnp.array([10, 20, 30])}},
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
                assert jnp.array_equal(
                    logs[key][subkey], loaded_logs[key][subkey]
                ), f"Mismatch in nested key: {key}->{subkey}"
        else:
            assert jnp.array_equal(
                logs[key], loaded_logs[key]
            ), f"Mismatch in key: {key}"


def test_load_nonexistent_file(tmp_path):
    path = str(tmp_path / "nonexistent.pkl")
    with pytest.raises(FileNotFoundError):
        load(path)


def test_save_invalid_path(logs):
    with pytest.raises(Exception):
        save(logs, "/invalid_path/test_logs.pkl")


def test_load_invalid_file(tmp_path):
    path = str(tmp_path / "invalid_file.txt")
    with open(path, "w") as f:
        f.write("This is not a valid pickle file.")
    with pytest.raises(Exception):
        load(path)


def test_save_empty_logs(tmp_path):
    empty_logs = {}
    path = str(tmp_path / "empty_logs.pkl")
    save(empty_logs, path)
    loaded_logs = load(path)
    assert loaded_logs == empty_logs, "Loaded logs should be empty dictionary"
