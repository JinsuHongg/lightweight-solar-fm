from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import numcodecs
import zarr
from xarray.coding import times
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view


def linear_interpolation(data):
    x = np.arange(len(data))
    mask = ~np.isnan(data)
    if mask.sum() == 0:
        return data
    interp_data = np.interp(x, x[mask], data[mask])
    # Clip values to be slightly above the 1e-9 threshold
    return np.maximum(interp_data, 1.0001e-9)


def main(
    file_paths: list,
    zarr_path: Path,
    window_hours: int,
    step_hours: int,
    time_filters: dict | None = None,
):
    """
    Preprocess XRS data into Zarr format.

    Args:
        file_paths: List of paths to XRS NC files.
        zarr_path: Output Zarr path.
        window_hours: Window size in hours.
        step_hours: Step size in hours.
        time_filters: Dict mapping satellite suffix (e.g., "g15") to
            (start_date, end_date) tuple. None means full range.
    """
    window_size_mins = window_hours * 60
    step_size_mins = step_hours * 60
    minute_offsets = np.arange(window_size_mins)

    compressor = numcodecs.Blosc(
        cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE
    )
    is_first_write = True

    buffer_soft = None
    buffer_hard = None
    buffer_time = None

    for file_idx, input_xrs_path in enumerate(file_paths):
        logger.info(
            f"Processing file {file_idx + 1}/{len(file_paths)}: {input_xrs_path.name}"
        )

        with xr.open_dataset(input_xrs_path) as ds:
            time = ds["time"].values
            hard = ds["xrsa_flux"].values
            soft = ds["xrsb_flux"].values

        # Extract satellite suffix (e.g., g15, g16, g18) from filename
        sat_suffix = None
        for suffix in ["g15", "g16", "g18"]:
            if suffix in input_xrs_path.name:
                sat_suffix = suffix
                break

        # Apply time filter if specified
        if time_filters is not None and sat_suffix is not None and sat_suffix in time_filters:
            start_date, end_date = time_filters[sat_suffix]
            time_pd = pd.to_datetime(time)

            if start_date is not None:
                start_dt = pd.to_datetime(start_date)
                mask_start = time_pd >= start_dt
                time = time[mask_start]
                hard = hard[mask_start]
                soft = soft[mask_start]
                time_pd = pd.to_datetime(time)
                logger.info(f"Filtered to start >= {start_date}")

            if end_date is not None:
                end_dt = pd.to_datetime(end_date)
                mask_end = time_pd <= end_dt
                time = time[mask_end]
                hard = hard[mask_end]
                soft = soft[mask_end]
                logger.info(f"Filtered to end <= {end_date}")

        soft_outlier_mask = soft <= 1e-9
        soft[soft_outlier_mask] = np.nan

        hard_outlier_mask = hard <= 1e-9
        hard[hard_outlier_mask] = np.nan

        soft_interp = linear_interpolation(soft)
        hard_interp = linear_interpolation(hard)

        if buffer_soft is not None:
            soft_interp = np.concatenate([buffer_soft, soft_interp])
            hard_interp = np.concatenate([buffer_hard, hard_interp])
            time = np.concatenate([buffer_time, time])
            logger.info("Successfully bridged data from previous year.")

        soft_windows = sliding_window_view(soft_interp, window_shape=window_size_mins)
        hard_windows = sliding_window_view(hard_interp, window_shape=window_size_mins)
        time_windows = sliding_window_view(time, window_shape=window_size_mins)

        buffer_soft = soft_interp[-window_size_mins:]
        buffer_hard = hard_interp[-window_size_mins:]
        buffer_time = time[-window_size_mins:]

        # target_t_times stores END of each window - add 1 minute to get the correct hour marker
        # Window ending at 23:59 becomes 00:00 of the NEXT day
        target_t_times = time_windows[:, -1]

        target_pd = pd.to_datetime(target_t_times)
        # Add 1 minute to get the correct hour (23:59 + 1min = 00:00 of next day)
        target_pd_corrected = target_pd + pd.Timedelta(minutes=1)
        valid_indices = np.where(target_pd_corrected.minute == 0)[0]

        if len(valid_indices) == 0:
            logger.warning("No valid on-the-hour targets found in this file.")
            continue

        first_valid = valid_indices[0]

        # Step by 60 minutes
        soft_windows = soft_windows[first_valid::step_size_mins]
        hard_windows = hard_windows[first_valid::step_size_mins]
        
        # Get raw timestamps, then add 1 minute to get correct hour (window ending at 23:59 -> 00:00 next hour)
        aligned_t_times_raw = target_t_times[first_valid::step_size_mins]
        aligned_t_times = pd.to_datetime(aligned_t_times_raw) + pd.Timedelta(minutes=1)

        # --- THE MERGE ---
        # Stack the two (N, 1440) arrays into a single (N, 1440, 2) array
        # axis=-1 puts the channels at the very end, perfect for PyTorch
        xray_windows = np.stack([soft_windows, hard_windows], axis=-1)

        # Package into the Dataset with the new "channel" dimension
        ds_out = xr.Dataset(
            {
                "xray": (["timestep", "minute_offset", "channel"], xray_windows),
            },
            coords={
                "timestep": aligned_t_times,
                "minute_offset": minute_offsets,
                "channel": ["soft", "hard"],
            },
        )

        # Deduplicate against existing Zarr data before appending
        if not is_first_write:
            existing_ds = xr.open_zarr(zarr_path, decode_times=False)
            existing_times = existing_ds["timestep"].values
            new_times = ds_out["timestep"].values

            # Find new unique times not already in the Zarr store
            existing_set = set(existing_times)
            mask_new = np.array([t not in existing_set for t in new_times])

            num_duplicates = (~mask_new).sum()
            if num_duplicates > 0:
                logger.warning(
                    f"Skipping {num_duplicates} duplicate timesteps already in Zarr store"
                )

            ds_out = ds_out.isel(timestep=mask_new)

        # NEW: Ensure no internal duplicates in the data to be written
        timestep_index = ds_out.timestep.to_index()
        if timestep_index.has_duplicates:
            num_internal_dups = timestep_index.duplicated().sum()
            logger.warning(
                f"Removing {num_internal_dups} internal duplicate timesteps in current chunk"
            )
            _, unique_idx = np.unique(ds_out.timestep.values, return_index=True)
            ds_out = ds_out.isel(timestep=unique_idx)

# Get reference time from existing zarr to maintain consistent units
        if not is_first_write:
            existing_units = existing_ds.timestep.attrs.get(
                "units", "hours since 2010-04-08 00:00:00"
            )
            existing_calendar = existing_ds.timestep.attrs.get(
                "calendar", "proleptic_gregorian"
            )
        else:
            existing_units = "hours since 2010-04-08 00:00:00"
            existing_calendar = "proleptic_gregorian"

        # Write or Append to Zarr
        encoding = {
            "timestep": {
                "dtype": "float64",  # Use float64 to avoid precision issues with large timestamps
                "units": existing_units,
                "calendar": existing_calendar,
            },
        }

        if is_first_write:
            # Add compressor and chunks for the first write
            encoding["xray"] = {
                "compressor": compressor,
                "chunks": (500, window_size_mins, 2),
            }
            ds_out.to_zarr(zarr_path, mode="w", encoding=encoding)
            is_first_write = False
            logger.info("Created new Zarr store and saved first year.")
        else:
            if len(ds_out["timestep"]) > 0:
                # Do not provide encoding for existing variables on append
                ds_out.to_zarr(zarr_path, append_dim="timestep")
                logger.info(
                    f"Appended {len(ds_out['timestep'])} timesteps to Zarr store."
                )
            else:
                logger.info("No new timesteps to append.")

    xr.open_zarr(zarr_path, decode_times=False)
    zarr.consolidate_metadata(zarr_path)

    logger.success("Consolidated zarr store successfully!")

    logger.success("Finished successfully! All files processed and seamlessly merged.")


if __name__ == "__main__":
    window_size = 24
    step_size = 1

    data_dir = Path("/media/jhong90/storage/surya/xrs")

    file_g15 = sorted(list(data_dir.glob("sci_xrsf-l2-avg1m_g15_*.nc")))
    file_g16 = sorted(list(data_dir.glob("sci_xrsf-l2-avg1m_g16_*.nc")))
    file_g18 = sorted(list(data_dir.glob("sci_xrsf-l2-avg1m_g18_*.nc")))

    file_list = file_g15 + file_g16 + file_g18

    time_filters = {
        "g15": ("2010-04-07", "2017-02-06"),
        "g16": (None, None),
        "g18": ("2025-04-07", "2025-12-31"),
    }

    zarr_target = Path("./data/xrs_24hour_slices_v2.zarr")

    main(file_list, zarr_target, window_size, step_size, time_filters)
