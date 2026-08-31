import pytest
from pathlib import Path
from bmi_troute import bmi_troute
from bmi_DAforcing import bmi_DAforcing

from test import temporarily_change_dir

@pytest.fixture
def sample_config():
    """Create a minimal sample config file for testing."""
    return Path(__file__).parents[1] / "LowerColorado_TX_v4/test_AnA_V4_HYFeature.yaml"

@pytest.fixture
def initialized_model(sample_config):
    """Fixture providing an initialized model for tests."""
    with temporarily_change_dir(sample_config.parent):
        model = bmi_troute()
        model.initialize(str(sample_config))
        yield model
        model.finalize()

@pytest.fixture
def DAforcing(sample_config):
    """Fixture providing an initialized model for tests."""
    with temporarily_change_dir(sample_config.parent):
        DAforcing = bmi_DAforcing()
        DAforcing.initialize(bmi_cfg_file=str(sample_config))
        yield DAforcing
        DAforcing.finalize()


