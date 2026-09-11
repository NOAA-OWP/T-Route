import pytest
from pathlib import Path

from test import temporarily_change_dir

# bmi_troute / bmi_DAforcing are not part of this repo's package layout (the BMI
# model lives in troute_nwm_bmi), so importing them at module scope made conftest
# collection fail and took EVERY test in this directory down with it, including
# ones that never touch them. Imported inside the fixtures instead: a test that
# actually requests these fixtures still fails loudly, the rest now collect.

@pytest.fixture
def sample_config():
    """Create a minimal sample config file for testing."""
    return Path(__file__).parents[1] / "LowerColorado_TX_v4/test_AnA_V4_HYFeature.yaml"

@pytest.fixture
def initialized_model(sample_config):
    """Fixture providing an initialized model for tests."""
    from bmi_troute import bmi_troute

    with temporarily_change_dir(sample_config.parent):
        model = bmi_troute()
        model.initialize(str(sample_config))
        yield model
        model.finalize()

@pytest.fixture
def DAforcing(sample_config):
    """Fixture providing an initialized model for tests."""
    from bmi_DAforcing import bmi_DAforcing

    with temporarily_change_dir(sample_config.parent):
        DAforcing = bmi_DAforcing()
        DAforcing.initialize(bmi_cfg_file=str(sample_config))
        yield DAforcing
        DAforcing.finalize()


