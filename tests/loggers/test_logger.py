from abc import ABC

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
