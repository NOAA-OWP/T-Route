from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt

def plot_reach(data_dir: str, reach_id: int, out_path: Path | str) -> None:
    data_dir = Path(data_dir)
    tds = [xr.open_dataset(p, engine="netcdf4") for p in sorted(data_dir.glob("*.nc"))]
    tds = xr.concat(tds, dim="time")
    df = tds.sel(feature_id=reach_id)["flow"].to_pandas()

    peak_idx = df.idxmax()
    peak_val = df.max()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_facecolor("whitesmoke")
    fig.patch.set_facecolor("whitesmoke")
    ax.plot(df.index, df.values, color="black", label=f"Reach {reach_id}")
    ax.plot(peak_idx, peak_val, "o", color="white", markeredgecolor="crimson", markeredgewidth=1.8, markersize=8, zorder=5)
    ax.annotate(
        f"Peak: {peak_val:.2f}",
        xy=(peak_idx, peak_val),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center",
        color="crimson",
        fontsize=9,
        fontweight=700,
    )
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
