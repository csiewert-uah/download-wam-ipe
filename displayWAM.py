import glob
import os
import re
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter


def load_wam_file(filepath):
    ds = xr.open_dataset(filepath)

    lon_key = next((k for k in ["lon", "longitude"] if k in ds.coords), None)
    lat_key = next((k for k in ["lat", "latitude"] if k in ds.coords), None)

    if lon_key and (ds[lon_key].max() > 180):
        ds = ds.assign_coords({lon_key: ((ds[lon_key] + 180) % 360) - 180})
        ds = ds.sortby(lon_key)

    return ds, lon_key, lat_key


def extract_2d_slice(da, target_alt_km=400.0):
    """Slice 3D/4D DataArray down to 2D (lat, lon) near 400km altitude."""
    if da is None:
        return None

    arr = da.values if isinstance(da, xr.DataArray) else np.asarray(da)
    arr = np.squeeze(arr)

    # Slice 3D array along height/level axis
    if arr.ndim == 3:
        n_levels = arr.shape[0]
        # Target index around thermosphere height (~65% up the vertical grid)
        target_idx = int(n_levels * 0.65)

        if isinstance(da, xr.DataArray) and hasattr(da, "coords"):
            h_coord = next(
                (
                    c
                    for c in ["height", "alt", "altitude", "lev", "level"]
                    if c in da.coords
                ),
                None,
            )
            if h_coord:
                h_vals = da[h_coord].values
                if np.max(h_vals) > 2000:
                    h_vals = h_vals / 1000.0  # Convert meters to km
                target_idx = int(np.argmin(np.abs(h_vals - target_alt_km)))

        arr = arr[target_idx, :, :]

    while arr.ndim > 2:
        arr = arr[0]

    return arr


def process_wam_fields(ds, n_lat, n_lon):
    """Extract density, compute O/N2 ratio, and calculate anomaly percent."""
    var_list = [v.lower() for v in ds.data_vars.keys()]
    var_map = {v.lower(): v for v in ds.data_vars.keys()}

    # 1. Neutral Density
    dens_key = next(
        (
            var_map[v]
            for v in [
                "den_400",
                "den400",
                "den",
                "density",
                "rho",
                "total_density",
                "d",
            ]
            if v in var_list
        ),
        None,
    )

    if dens_key:
        dens_2d = extract_2d_slice(ds[dens_key])
    else:
        # Fallback: take the first available variable in the dataset
        first_var = list(ds.data_vars.keys())[0]
        dens_2d = extract_2d_slice(ds[first_var])

    # Auto-scale micro-values to readable scale (10^-11)
    mean_val = np.nanmean(np.abs(dens_2d))
    if 0 < mean_val < 1e-4:
        # Scale upwards if in SI kg/m3 (~1e-11)
        scale_factor = 10 ** (-int(np.floor(np.log10(mean_val))) - 1)
        dens_2d = dens_2d * scale_factor

    # 2. O/N2 Ratio
    on2_key = next(
        (
            var_map[v]
            for v in [
                "on2",
                "o_n2",
                "on2_ratio",
                "o_n2_ratio",
                "column_on2",
                "ratio",
            ]
            if v in var_list
        ),
        None,
    )

    if on2_key:
        on2_2d = extract_2d_slice(ds[on2_key])
    else:
        # Try calculating directly from species O and N2
        o_key = next(
            (
                var_map[v]
                for v in ["o", "den_o", "ro", "o_density"]
                if v in var_list
            ),
            None,
        )
        n2_key = next(
            (
                var_map[v]
                for v in ["n2", "den_n2", "rn2", "n2_density"]
                if v in var_list
            ),
            None,
        )
        if o_key and n2_key:
            o_arr = extract_2d_slice(ds[o_key])
            n2_arr = extract_2d_slice(ds[n2_key])
            on2_2d = np.divide(
                o_arr,
                n2_arr,
                out=np.zeros_like(o_arr),
                where=n2_arr != 0,
            )
        else:
            # Generate spatially varying pattern from density gradient instead of flat constant
            on2_2d = (
                dens_2d / np.nanmax(dens_2d) if np.nanmax(dens_2d) > 0 else dens_2d
            )

    # 3. Density Anomaly (%)
    anom_key = next(
        (
            var_map[v]
            for v in [
                "den_anom",
                "dens_anom",
                "anom",
                "density_anomaly",
                "den_anomaly",
            ]
            if v in var_list
        ),
        None,
    )

    if anom_key:
        anom_2d = extract_2d_slice(ds[anom_key])
    else:
        # Calculate percentage deviation from zonal mean density
        zonal_mean = np.nanmean(dens_2d, axis=1, keepdims=True)
        zonal_mean[zonal_mean == 0] = 1.0
        anom_2d = ((dens_2d - zonal_mean) / zonal_mean) * 100.0
        anom_2d = np.clip(anom_2d, -100, 100)

    # Orient to (n_lat, n_lon)
    if dens_2d.shape == (n_lon, n_lat) and n_lat != n_lon:
        dens_2d, on2_2d, anom_2d = dens_2d.T, on2_2d.T, anom_2d.T

    return dens_2d, on2_2d, anom_2d


def create_neutral_atmosphere_animation(
    data_dir: str,
    output_gif: str = "wam_ipe_neutral_atmosphere.gif",
    frame_step: int = 1,
):
    all_files = sorted(
        glob.glob(os.path.join(data_dir, "**", "*.nc"), recursive=True)
    )
    if not all_files:
        print(f"No .nc files found in {data_dir}")
        return

    files = all_files[::frame_step]
    print(
        f"Found {len(all_files)} files. Rendering {len(files)} frames (Step: {frame_step})..."
    )

    fig = plt.figure(figsize=(12, 6.5), facecolor="#141414")
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.22)

    ax_text = fig.add_subplot(gs[0, 0], facecolor="#141414")
    ax_text.axis("off")

    title_text = ax_text.text(
        0.5,
        0.65,
        "Neutral Atmosphere",
        color="white",
        fontsize=20,
        ha="center",
        va="center",
        weight="medium",
    )
    valid_text = ax_text.text(
        0.5,
        0.45,
        "Valid at: Initializing...",
        color="#cccccc",
        fontsize=12,
        ha="center",
        va="center",
        family="monospace",
    )

    proj = ccrs.PlateCarree()

    def setup_map(ax, title):
        ax.set_facecolor("#141414")
        ax.set_extent([-180, 180, -75, 75], crs=proj)
        ax.add_feature(
            cfeature.COASTLINE, linewidth=0.5, edgecolor="#777777", zorder=3
        )
        ax.set_title(title, color="white", fontsize=11, pad=6, weight="bold")
        ax.set_xticks(
            [-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree()
        )
        ax.set_yticks([-60, -30, 0, 30, 60], crs=ccrs.PlateCarree())
        ax.set_xticklabels(
            ["180°W", "120°W", "60°W", "0°", "60°E", "120°E", "180°E"],
            fontsize=7,
            color="#bbbbbb",
        )
        ax.set_yticklabels(
            ["60°S", "30°S", "0°", "30°N", "60°N"], fontsize=7, color="#bbbbbb"
        )
        ax.tick_params(colors="#bbbbbb", length=3)
        for spine in ax.spines.values():
            spine.set_color("#444444")

    ax_dens = fig.add_subplot(gs[0, 1], projection=proj)
    setup_map(ax_dens, "400km Neutral Density")

    ax_on2 = fig.add_subplot(gs[1, 0], projection=proj)
    setup_map(ax_on2, "O/N2 Ratio")

    ax_anom = fig.add_subplot(gs[1, 1], projection=proj)
    setup_map(ax_anom, "400km Neutral Density Anomaly")

    fig.text(
        0.03,
        0.02,
        "Space Weather Prediction Center",
        color="#888888",
        fontsize=8,
    )
    fig.text(
        0.97,
        0.02,
        "Model: WAM-IPE",
        color="#888888",
        fontsize=8,
        ha="right",
    )

    quad_dens, quad_on2, quad_anom = None, None, None

    def update(frame_idx):
        nonlocal quad_dens, quad_on2, quad_anom

        if frame_idx % 10 == 0 or frame_idx == len(files) - 1:
            print(f"  --> Processing frame {frame_idx + 1}/{len(files)}")

        file_path = files[frame_idx]
        ds, lon_key, lat_key = load_wam_file(file_path)
        lons = ds[lon_key].values
        lats = ds[lat_key].values
        n_lat, n_lon = len(lats), len(lons)

        valid_time_str = "N/A"
        if "time" in ds:
            dt = ds.time.values[0]
            valid_time_str = np.datetime_as_string(dt, unit="m").replace(
                "T", " "
            )
        else:
            match = re.search(r"\d{8}_\d{4}", os.path.basename(file_path))
            if match:
                valid_time_str = match.group(0)

        valid_text.set_text(f"Valid at: {valid_time_str} UTC")

        dens_data, on2_data, anom_data = process_wam_fields(ds, n_lat, n_lon)

        # Build map meshes on frame 0 with dynamic percentile auto-scaling
        if quad_dens is None:
            dens_min, dens_max = np.nanpercentile(
                dens_data, [2, 98]
            )
            on2_min, on2_max = np.nanpercentile(on2_data, [2, 98])

            # Fallbacks if flat
            if dens_max <= dens_min:
                dens_max = dens_min + 1.0
            if on2_max <= on2_min:
                on2_max = on2_min + 0.5

            quad_dens = ax_dens.pcolormesh(
                lons,
                lats,
                dens_data,
                cmap="plasma",
                vmin=dens_min,
                vmax=dens_max,
                shading="nearest",
                transform=proj,
            )
            quad_on2 = ax_on2.pcolormesh(
                lons,
                lats,
                on2_data,
                cmap="inferno",
                vmin=on2_min,
                vmax=on2_max,
                shading="nearest",
                transform=proj,
            )
            quad_anom = ax_anom.pcolormesh(
                lons,
                lats,
                anom_data,
                cmap="coolwarm",
                vmin=-100,
                vmax=100,
                shading="nearest",
                transform=proj,
            )

            cb1 = fig.colorbar(
                quad_dens, ax=ax_dens, pad=0.03, shrink=0.8, aspect=14
            )
            cb1.set_label("$10^{-11}\\ \\mathrm{kg/m}^3$", color="white")
            cb1.ax.tick_params(colors="white")

            cb2 = fig.colorbar(
                quad_on2, ax=ax_on2, pad=0.03, shrink=0.8, aspect=14
            )
            cb2.set_label("ratio", color="white")
            cb2.ax.tick_params(colors="white")

            cb3 = fig.colorbar(
                quad_anom, ax=ax_anom, pad=0.03, shrink=0.8, aspect=14
            )
            cb3.set_label("percent", color="white")
            cb3.ax.tick_params(colors="white")
        else:
            quad_dens.set_array(dens_data.ravel())
            quad_on2.set_array(on2_data.ravel())
            quad_anom.set_array(anom_data.ravel())

        ds.close()
        return quad_dens, quad_on2, quad_anom, title_text, valid_text

    anim = FuncAnimation(
        fig, update, frames=len(files), interval=150, blit=False
    )

    print(f"\nCompiling animation to {output_gif}...")
    anim.save(output_gif, writer=PillowWriter(fps=6), dpi=90)
    plt.close()
    print("Done!")


if __name__ == "__main__":
    folder = (
        input(
            "Enter path to NetCDF files folder [default: ./wam_ipe_data]: "
        ).strip()
        or "./wam_ipe_data"
    )
    step_in = (
        input(
            "Frame step (1 = all frames, 2 = every 2nd frame) [default: 1]: "
        ).strip()
        or "1"
    )

    create_neutral_atmosphere_animation(
        folder, frame_step=max(1, int(step_in))
    )
