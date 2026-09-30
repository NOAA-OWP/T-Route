from typing import Any, Dict

import pandas as pd
import pytest

@pytest.fixture
def hyfeature_network(hyfeatures_test_network: Dict[str, Any]) -> pd.DataFrame:
    cwd = hyfeatures_test_network["cwd"]
    return pd.read_parquet(cwd / "test/troute-network/data/_dataframe.parquet")
