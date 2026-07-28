from abc import ABC, abstractmethod

import pytest
from functions import *

import lox
from lox.loggers import Logger

functions = [
    f_id,
    f_add,
    f_scan,
]


class TestLogger(ABC):
    @abstractmethod
    def logger(self) -> Logger:
        pass

    @pytest.fixture(params=[0, 1, 2])
    def key(self, request):
        seed = request.param
        return jax.random.key(seed)

    @pytest.fixture(params=functions)
    def f(self, request):
        return request.param

    @pytest.fixture(params=[(4,), (2, 3), (5, 2)])
    def x(self, request):
        shape = request.param
        return jax.random.uniform(jax.random.key(0), shape=shape)

    def test_spool(self, logger, key, f, x):
        logger_state = logger.init(key)
        _ = logger.spool(f, logger_state)(x)

    def test_tap(self, logger, key, f, x):
        logger_state = logger.init(key)
        _ = logger.tap(f, logger_state)(x)

    def test_log(self, logger, key, f, x):
        logger_state = logger.init(key)
        _, logs = lox.spool(f)(x)
        logger.log(logger_state, logs)

    def test_spool_argnames(self, logger, key, x):
        logger_state = logger.init(key)
        collected = []
        logger.callback = lambda logger_state, logs: collected.append(logs)
        logger.spool(_f_ab_train_c_eval, logger_state, argnames=["a"])(x)
        keys = set()
        for logs in collected:
            keys |= set(logs.keys())
        assert keys == {"a"}

    def test_spool_tags(self, logger, key, x):
        logger_state = logger.init(key)
        collected = []
        logger.callback = lambda logger_state, logs: collected.append(logs)
        logger.spool(_f_ab_train_c_eval, logger_state, tags=["train"])(x)
        keys = set()
        for logs in collected:
            keys |= set(logs.keys())
        assert keys == {"a", "b"}

    def test_tap_tags(self, logger, key, x):
        logger_state = logger.init(key)
        collected = []
        logger.callback = lambda logger_state, logs: collected.append(logs)
        logger.tap(_f_ab_train_c_eval, logger_state, tags=["train"])(x)
        keys = set()
        for logs in collected:
            keys |= set(logs.keys())
        assert keys == {"a", "b"}


def _f_ab_train_c_eval(x):
    lox.log({"a": x, "b": x}, tags=("train",))
    lox.log({"c": x}, tags=("eval",))
    return x + 1
