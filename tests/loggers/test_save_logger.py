import pytest
from test_logger import TestLogger

from lox.loggers import SaveLogger


class TestSaveLogger(TestLogger):
    @pytest.fixture
    def logger(self, tmp_path):
        return SaveLogger(path=str(tmp_path))
