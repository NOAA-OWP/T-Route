# Test Directory Structure

This directory is meant for the following purposes:
- Integration testing of individual functions/class objects within T-Route
- Hosting example data to run full-length test runs

## Quick Start

```bash
# Run all tests
pytest test/

# Run tests from a specific category
pytest test/troute-config/
pytest test/troute-network/
pytest test/troute-nwm/
pytest test/troute-bmi/
```

## Test Categories

### Config Tests (`test/troute-config/`)

**Usage:**
- Validating existing Troute example configs
- Testing the Pydantic Config Validations

### Network Tests (`test/troute-network/`)

**Usage:**
- Using YAML files from `NHD` and `HYFeatures` in order to test network creation

### NWM Tests (`test/troute-nwm/`)

**Usage:**
- end to end tests of individual functions within the following functions:
  - `main_v03()`
  - `main_v04()`

### BMI Tests (`test/troute-bmi/`)

**Usage:**
- Integration tests for the BMI NWM implementation of T-Route

### End to End Examples:
##### `test/LowerColorado_TX/`
##### `test/LowerColorado_TX_v4/`
##### `test/unit_test_hyfeature/`

## Best Practices

1. Always use the appropriate fixture for your test case to minimize setup code
2. Use `temporarily_change_dir` context manager when working with paths to ensure you're able to run integration tests successfully
3. Check fixture contents before using them in tests
4. Use fixture combinations when testing complex interactions

## Adding New Fixtures

When adding new fixtures:
1. Follow the existing naming convention
2. Document the fixture structure and purpose
3. Include example usage
4. Add any necessary dependencies
5. Update this README with new end to end test docs or examples

## Adding New Tests

1. Create new test files in the appropriate category directory
2. Use the naming convention `test_*.py` for test files
3. Document any new fixtures in the relevant conftest.py
