import shutil

import pandas as pd
import yaml
from troute.config import Config
from troute_rnr import format, nwps
from troute_rnr.settings import Settings


def test_format_config(
    mock_rfc_inputs: nwps.ProcessedData,
    mock_settings: Settings,
    mock_flows: pd.DataFrame,
    mock_restart_file: pd.DataFrame,
) -> None:
    """Tests to make sure all t-route inputs are formatted correctly

    Parameter
    ---------
    mock_rfc_inputs: nwps.ProcessedData
        The preprocessed RFC inputs
    mock_settings: Settings
        The settings for the t-route formatting
    mock_flows: pd.DataFrame
        The correct flow values
    mock_restart_file: pd.DataFrame
        The correct restart file
    """
    yaml_file_path, tmp_flow_files_path = format.format_config(mock_rfc_inputs, mock_settings)

    with open(yaml_file_path) as custom_file:
        data = yaml.load(custom_file, Loader=yaml.SafeLoader)

    # Testing to make sure config will validate in strict mode
    _ = Config.with_strict_mode(**data)

    # Testing the flows
    first_flows_df = pd.read_csv(tmp_flow_files_path / "202505052100.CHRTOUT_DOMAIN1.csv")
    pd.testing.assert_frame_equal(
        first_flows_df,
        mock_flows,
        check_dtype=True,
        check_index_type=False,
        check_column_type=False,
        rtol=1e-5,
    )

    # Testing the restart file
    restart_df = pd.read_pickle(mock_settings.restart_path / f"{mock_rfc_inputs.lid}/2025-05-05_21:00.pkl")
    pd.testing.assert_frame_equal(
        restart_df,
        mock_restart_file,
        check_dtype=True,
        check_index_type=False,
        check_column_type=False,
        rtol=1e-5,
    )

    # Deleting test files
    yaml_file_path.unlink()
    shutil.rmtree(tmp_flow_files_path)
    shutil.rmtree(mock_settings.restart_path / mock_rfc_inputs.lid)
