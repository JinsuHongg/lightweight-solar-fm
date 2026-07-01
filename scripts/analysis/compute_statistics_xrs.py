import argparse
import os

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from omegaconf import OmegaConf
from xarray.coding.times import CFDatetimeCoder


def compute_statistics(
    zarr_path: str,
    index_path: str | None = None,
    output_path: str | None = None,
):
    """
    Compute statistics for XRS data in a Zarr dataset.

    Args:
        zarr_path (str): Path to the Zarr dataset.
        index_path (str, optional): Path to the index CSV file (containing timestamps) to filter data. Defaults to None.
        output_path (str, optional): Path to save the statistics as a YAML file. Defaults to None.
    """
    logger.info(f"Opening Zarr dataset at {zarr_path}")

    # Open with automatic cftime decoding
    time_coder = CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(zarr_path, engine="zarr", chunks="auto", decode_times=time_coder)

    if "xray" not in ds.data_vars:
        logger.error("Variable 'xray' not found in the Zarr dataset.")
        return

    xray = ds["xray"]
    logger.info(f"Found variable 'xray' with shape {xray.shape} and dims {xray.dims}")

    if index_path is not None:
        logger.info(f"Loading index from {index_path} to filter data")
        try:
            index_df = pd.read_csv(index_path)
            index_df["timestamp"] = pd.to_datetime(
                index_df["timestamp"]
            ).values.astype("datetime64[ns]")
            index_df.set_index("timestamp", inplace=True)
            index_df.sort_index(inplace=True)

            # Drop duplicate timestamps from CSV
            if index_df.index.has_duplicates:
                num_dups = index_df.index.duplicated().sum()
                logger.warning(
                    f"Index CSV has {num_dups} duplicate timestamps, keeping first occurrence"
                )
                index_df = index_df[~index_df.index.duplicated(keep="first")]
            selected_timestamps = index_df.index

            # Handle duplicates in the dataset by keeping the first occurrence
            if ds.indexes["timestep"].has_duplicates:
                num_dups = ds.indexes["timestep"].duplicated().sum()
                logger.warning(
                    f"Dataset 'timestep' has {num_dups} duplicate values, keeping first occurrence"
                )
                ds = ds.sel(timestep=~ds.indexes["timestep"].duplicated())

            # Convert pandas datetime64 index to cftime objects to match dataset
            calendar = ds.timestep.encoding.get("calendar", "proleptic_gregorian")
            selected_cftimes = [
                xr.coding.times.cftime.datetime(
                    t.year,
                    t.month,
                    t.day,
                    t.hour,
                    t.minute,
                    t.second,
                    calendar=calendar,
                )
                for t in selected_timestamps
            ]

            logger.info(
                f"Reindexing to find {len(selected_cftimes)} common timestamps..."
            )
            # Use reindex with a tolerance to find nearest timestamps
            ds_filtered = ds.reindex(
                {"timestep": selected_cftimes},
                method="nearest",
                tolerance=pd.Timedelta("1 minute"),
            )

            # Drop any NaNs that resulted from timestamps with no close match
            ds_filtered = ds_filtered.dropna(dim="timestep")

            if len(ds_filtered.timestep) == 0:
                logger.error("No common timestamps found after reindexing.")
                return

            logger.info(
                f"Found {len(ds_filtered.timestep)} common timestamps after reindexing."
            )
            xray = ds_filtered["xray"]

            logger.info(f"After filtering, data shape: {xray.shape}")
        except Exception as e:
            logger.error(f"Failed to load or apply index filter: {e}")
            return

    eps = 1e-10

    stats = {}
    for channel in ["soft", "hard"]:
        arr = xray.sel(channel=channel).data

        arr_log = da.log10(da.clip(arr, eps, None))

        logger.info(f"Computing stats for {channel}...")
        mean = float(arr_log.mean().compute())
        std = float(arr_log.std().compute())
        mn = float(arr_log.min().compute())
        mx = float(arr_log.max().compute())

        stats[channel] = {
            "mean": mean,
            "std": std,
            "min": mn,
            "max": mx,
        }
        logger.info(
            f"  {channel}: mean={mean:.4f}, std={std:.4f}, min={mn:.4f}, max={mx:.4f}"
        )

    output_path = output_path or "xrs_stat.yaml"
    logger.info(f"Saving statistics to {output_path}")
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        OmegaConf.save(OmegaConf.create(stats), output_path)
        logger.success(f"Successfully saved statistics to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save statistics to {output_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute statistics of XRS Zarr dataset."
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="./data/xrs_24hour_slices_v2.zarr",
        help="Path to the Zarr dataset.",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="./data/pretrain/train.csv",
        help="Path to the index CSV file (containing timestamps) to filter data for statistics computation.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/xrs_stat_train.yaml",
        help="Path to save the computed statistics in YAML format.",
    )
    args = parser.parse_args()

    compute_statistics(args.zarr_path, args.index_path, args.output_path)
