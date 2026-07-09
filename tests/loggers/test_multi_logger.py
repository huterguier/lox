import pytest
from test_logger import TestLogger

from lox.loggers import MultiLogger, SaveLogger


class TestMultiLogger(TestLogger):
    @pytest.fixture
    def logger(self, tmp_path):
        return MultiLogger(
            SaveLogger(path=str(tmp_path / "a")),
            SaveLogger(path=str(tmp_path / "b")),
        )
