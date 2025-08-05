import numpy as np
import pandas as pd


def write_flow_files(mapped_feature_id, reach, output_path):
    """Writes csv flow files for each rfc point"""
    times = reach.times
    filtered_data = np.array(reach.secondary_forecast)
    for idx, time in enumerate(times):
        formatted_time = time.strftime("%Y%m%d%H%M")
        _df = pd.DataFrame(
            {
                "feature_id": [mapped_feature_id],
                formatted_time: [filtered_data[idx]],
            }
        )
        file_path = output_path / f"{formatted_time}.CHRTOUT_DOMAIN1.csv"
        _df.to_csv(file_path, index=False)
