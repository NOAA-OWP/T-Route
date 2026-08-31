"""Run a t-route config file via BMI."""
import argparse
import os
from pathlib import Path
import sys
import glob
from datetime import datetime
import numpy as np
import pandas as pd

from troute_nwm_bmi.troute_bmi import BmiTroute


def run_bmi(config_path: str):
    # Emulate running from config directory
    config_path: Path = Path(config_path).resolve()
    os.chdir(config_path.parent)
    config_path = config_path.relative_to(config_path.parent)

    # Initialize BMI model
    model = BmiTroute()
    model.initialize(config_path)

    dt = model.get_time_step()  # 300s

    # Collect and sort forcing files by timestamp
    forcing_dir = model._model._network.forcing_parameters["qlat_input_folder"]
    forcing_pattern = model._model._network.forcing_parameters["qlat_file_pattern_filter"]
    forcing_files = sorted(glob.glob(os.path.join(forcing_dir, forcing_pattern)))
    if not forcing_files:
        print(f"No forcing files found in {forcing_dir}/{forcing_pattern}")
        sys.exit(1)

    print(f"Found {len(forcing_files)} forcing files, dt={dt}s")

    # Read first file to get IDs (all files share the same feature_id index)
    first_df = pd.read_csv(forcing_files[0]).set_index("feature_id")
    feature_ids = np.array(first_df.index, dtype=np.intc)

    # Set the IDs (constant across all timesteps)
    model.set_value("catchment_water_source__id", feature_ids)

    # Build forcing data
    flow_values = pd.concat([pd.read_csv(i).set_index("feature_id") for i in forcing_files], axis=1)
    flow_values = flow_values.values.flatten(order="F")
    model.set_value("catchment_water_source__volume_flow_rate", flow_values)
    model.update_until(model._model.forcing_parameters["nts"])


    # Finalize triggers routing computation and output writing
    print("Running routing computation...")
    model.finalize()
    print("Done. Check output/ directory for results.")

def main():
    parser = argparse.ArgumentParser(
        description="Execute a t-route run from a config yaml using BMI."
    )

    parser.add_argument(
        "--config-file",
        help="Path to the config yaml for the run of interest.",
    )

    args = parser.parse_args()

    run_bmi(args.config_file)


if __name__ == "__main__":
    main()

