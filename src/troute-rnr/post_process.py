"""A function to format t-route outputs into the correct format for water.noaa.gov"""

import re
from datetime import datetime, timedelta

import pandas as pd
import xarray as xr
from troute_rnr.settings import Settings


def extract_timestamp_from_filename(filename: str) -> datetime | None:
    """Extract timestamp from filename like 'troute_output_202505061230.nc'

    Parameters
    ----------
    filename : str
        The filename to extract timestamp from (format: troute_output_YYYYMMDDHHMM.nc)

    Returns
    -------
    datetime | None
        The extracted datetime, or None if parsing fails
    """
    # Pattern to match troute_output_*.nc and extract the timestamp
    pattern = r"troute_output_(\d{12})\.nc"

    match = re.search(pattern, filename)
    if not match:
        return None
    timestamp_str = match.group(1)

    try:
        # Parse YYYYMMDDHHMM format
        year = int(timestamp_str[:4])
        month = int(timestamp_str[4:6])
        day = int(timestamp_str[6:8])
        hour = int(timestamp_str[8:10])
        minute = int(timestamp_str[10:12])
        return datetime(year, month, day, hour, minute)
    except (ValueError, IndexError) as e:
        print(f"Error parsing timestamp from {filename}: {e}")
        return None


def post_process(settings: Settings) -> None:
    """A function to post-process the T-Route outputs from RnR

    Parameter
    ---------
    settings: Settings
        The global RnR settings
    """
    data_dict = {
        "feature_id": [],
        "feature_id_str": [],
        "strm_order": [],
        "name": [],
        "state": [],
        "streamflow_cfs": [],
        "inherited_rfc_forecasts": [],
        "max_status": [],
        "reference_time": [],
        "update_time": [],
        "geom": [],
    }
    current_time = datetime.now()
    twenty_four_hours_ago = current_time - timedelta(hours=24)
    timestamp = current_time.strftime("%Y-%m-%d_%H:%M:%S")

    # Reading in the hydrofabric
    print("Reading the hydrofabric")
    flowpaths = pd.read_parquet(settings.data_dir / "parquet/flowpaths.parquet")
    flowpaths = flowpaths.set_index("id")
    write_file = False

    print("Opening all forecasts for times after the current timestep")
    for folder in settings.output_files_path.glob("*"):
        if folder.is_dir():
            for nc_file in folder.glob("*.nc"):
                file_timestamp = extract_timestamp_from_filename(nc_file.name)
                # Filter files created within the last 24 hours
                if (
                    file_timestamp and file_timestamp >= twenty_four_hours_ago
                ):  # Searches for files with timestamps within the past 24 hours
                    ds = xr.open_dataset(nc_file, engine="netcdf4")
                    write_file = True

                    # Find which feature_id and time index corresponds to the global max
                    global_max_flow = ds.flow.max()

                    max_location = (
                        ds.flow.where(ds.flow == global_max_flow)
                        .stack(flat_dim=["feature_id", "time"])
                        .dropna("flat_dim")
                    )
                    max_time_idx = max_location.time.values[0]
                    max_ds = ds.sel(time=max_time_idx)
                    catchments = [f"wb-{_id}" for _id in max_ds.feature_id.values]
                    filtered_flowpaths = flowpaths.loc[flowpaths.index.isin(catchments)]
                    data_dict["feature_id"].extend(
                        max_ds.feature_id.values
                    )  # Using the hydrofabric v2.2 IDs since there are many NHD feature IDs per hydrofabric catchment
                    data_dict["feature_id_str"].extend(catchments)
                    data_dict["strm_order"].extend([max_ds.attrs["stream_order"]] * len(catchments))
                    data_dict["name"].extend([max_ds.attrs["name"]] * len(catchments))
                    data_dict["state"].extend([max_ds.attrs["state"]] * len(catchments))
                    data_dict["max_status"].extend([max_ds.attrs["max_status"]] * len(catchments))
                    data_dict["reference_time"].extend(
                        [max_ds.attrs["file_reference_time"]] * len(catchments)
                    )
                    data_dict["update_time"].extend([timestamp] * len(catchments))
                    data_dict["streamflow_cfs"].extend(max_ds.flow.values * 35.3147)  # to cfs

                    total_miles = 0.0
                    miles_upstream = [total_miles]
                    # Flowpaths are pre-sorted by upstream to downstream
                    for i, (_, row) in enumerate(filtered_flowpaths.iterrows()):
                        if (
                            i != len(filtered_flowpaths) - 1
                        ):  # Skipping the last segment since its miles from upstream based on the upstream connection
                            total_miles += row["lengthkm"] * 0.621371  # converting km to miles
                            miles_upstream.append(total_miles)

                    data_dict["inherited_rfc_forecasts"].extend(
                        [
                            f"{max_ds.attrs['max_status']} issued {max_ds.attrs['file_reference_time']} at {max_ds.attrs['rfc_location']} ({max_ds.attrs['rfc_reach_id']} [order {max_ds.attrs['stream_order']}]) {miles} miles upstream"
                            for miles in miles_upstream
                        ]
                    )
                    data_dict["geom"].extend(filtered_flowpaths.geometry.values.tolist())
                    ds.close()

    if write_file:
        output_filename = f"output_inundation_{timestamp}.csv"
        output_path = settings.output_files_path / output_filename
        df = pd.DataFrame(data_dict)
        df.to_csv(output_path)
        print(f"Processing complete! Results saved to: {output_path}")
    else:
        print(
            "No new forecasts have been received. No file has been written. Please rerun RnR to get more recently routed flows"
        )


if __name__ == "__main__":
    post_process(Settings())
