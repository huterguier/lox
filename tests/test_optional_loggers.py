import pytest
from lox.loggers import ConsoleLogger, Logger, MultiLogger, SaveLogger
from lox.utils.optional import MissingDependency

def test_standard_loggers_importable():
    assert ConsoleLogger is not None
    assert Logger is not None
    assert MultiLogger is not None
    assert SaveLogger is not None

def test_wandb_logger_lazy_loading():
    # This should not raise an error even if wandb is missing
    from lox.loggers import WandbLogger
    
    # If wandb is missing, WandbLogger should be a MissingDependency instance
    # If wandb is present, it should be the real class.
    # We can't easily force it to be missing here without mocking,
    # but we can check the behavior if it IS a MissingDependency.
    
    if isinstance(WandbLogger, MissingDependency):
        with pytest.raises(ImportError, match="'WandbLogger' requires the 'wandb' package"):
            WandbLogger()
    else:
        # If it's the real class, it should be a subclass of Logger
        assert issubclass(WandbLogger, Logger)

def test_missing_dependency_stub():
    stub = MissingDependency("TestLogger", "test-package")
    with pytest.raises(ImportError, match="'TestLogger' requires the 'test-package' package"):
        stub()
    
    with pytest.raises(ImportError, match="'TestLogger' requires the 'test-package' package"):
        stub.some_method()
