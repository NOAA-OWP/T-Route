from troute_ewts.constants import (
    MODULE_NAME,
    LOG_MODULE_NAME_LEN,
)

def test_module_name_is_string():
    assert isinstance(MODULE_NAME, str)

def test_module_name_length_fits_field():
    assert len(MODULE_NAME) <= LOG_MODULE_NAME_LEN
