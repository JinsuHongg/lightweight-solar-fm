import argparse
import os

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from loguru import logger
from omegaconf import OmegaConf
from xarray.coding.times import CFDatetimeCoder


def compute_statistics(
    zarr_path: str,
    variable_name: str,
    index_path: str | None = None,
    output_path: str | None = None,
):
    """
    Compute statistics for a given variable in a Zarr dataset.

    Args:
        zarr_path (str): Path to the Zarr dataset.
        variable_name (str): Name of the variable to compute statistics on.
        index_path (str, optional): Path to the index CSV file (containing timestamps) to filter data. Defaults to None.
        output_path (str, optional): Path to save the statistics as a YAML file. Defaults to None.
    """
    logger.info(f"Opening Zarr store at {zarr_path}")

    store = zarr.open(zarr_path, mode="r")

    if not isinstance(store, zarr.hierarchy.Group):
        logger.error(
            f"Zarr store at {zarr_path} is not a Group. This script is designed to work with Zarr groups."
        )
        return

    available_arrays = list(store.array_keys())
    if variable_name not in available_arrays:
        logger.error(f"Variable '{variable_name}' not found in the Zarr group.")
        logger.info(f"Available arrays are: {available_arrays}")
        return

    zarr_array = store[variable_name]
    logger.info(
        f"Found variable '{variable_name}' with shape {zarr_array.shape} and {zarr_array.ndim} dimensions."
    )

    dims = None
    if variable_name == "images":
        if zarr_array.ndim == 4:
            dims = ["time", "y", "x", "channel"]
        elif zarr_array.ndim == 3:
            dims = ["time", "y", "x"]
    elif variable_name == "timestamps":
        if zarr_array.ndim == 1:
            dims = ["time"]

    if dims is None:
        logger.error(
            f"Could not infer dimension names for variable '{variable_name}' with {zarr_array.ndim} dimensions."
        )
        return

    logger.info(f"Assuming dimensions: {dims}")

    use_xarray = False
    ts_coords = None

    if "timestamps" in available_arrays:
        try:
            time_coder = CFDatetimeCoder(use_cftime=True)
            ds = xr.open_dataset(
                zarr_path, engine="zarr", chunks="auto", decode_times=time_coder
            )
            ts_coords = ds["time"].values
            use_xarray = True
            logger.info("Loaded timestamps with xarray (time metadata detected)")
        except Exception:
            logger.info("Time metadata not found, using raw zarr timestamps")
            ts_array = store["timestamps"]
            ts_coords = pd.to_datetime(ts_array[:], unit="s")

    dask_array = da.from_array(zarr_array, chunks=zarr_array.chunks)
    data_var = xr.DataArray(dask_array, dims=dims, name=variable_name)

    if index_path is not None:
        logger.info(f"Loading index from {index_path} to filter data")
        try:
            index_df = pd.read_csv(index_path)
            index_df["timestamp"] = pd.to_datetime(index_df["timestamp"]).values.astype(
                "datetime64[ns]"
            )
            index_df.set_index("timestamp", inplace=True)
            index_df.sort_index(inplace=True)

            index_df = index_df[~index_df.index.duplicated(keep="first")]
            selected_timestamps = index_df.index

            time_dim = dims[0]
            logger.info(
                f"Filtering data by {len(selected_timestamps)} timestamps using dimension '{time_dim}'"
            )

            if use_xarray:
                ts_index = pd.DatetimeIndex(ts_coords)

                matched_indices = []
                tolerance = pd.Timedelta("1 minute")
                for ts in selected_timestamps:
                    diffs = (ts_index - ts).to_numpy()
                    matches = np.where(np.abs(diffs) <= tolerance)[0]
                    if len(matches) > 0:
                        matched_indices.append(matches[0])
            else:
                if ts_coords is not None:
                    ts_index = pd.DatetimeIndex(ts_coords)

                    matched_indices = []
                    tolerance = pd.Timedelta("1 minute")
                    for ts in selected_timestamps:
                        diffs = (ts_index - ts).to_numpy()
                        matches = np.where(np.abs(diffs) <= tolerance)[0]
                        if len(matches) > 0:
                            matched_indices.append(matches[0])
                else:
                    matched_indices = list(range(zarr_array.shape[0]))

            if not matched_indices:
                logger.error("No common timestamps found after matching.")
                return

            matched_indices = np.array(matched_indices)
            logger.info(
                f"Found {len(matched_indices)} common timestamps after matching."
            )

            data_var = data_var.isel({time_dim: list(matched_indices)})

            logger.info(f"After filtering, data shape: {data_var.shape}")
        except Exception as e:
            logger.error(f"Failed to load or apply index filter: {e}")
            return

    logger.info(f"Computing statistics for variable '{variable_name}'")

    dask_array = data_var.data
    if not isinstance(dask_array, da.Array):
        dask_array = da.from_array(dask_array, chunks="auto")

    with dask.config.set(scheduler="threads"):
        mean_val = dask_array.mean().compute()
        std_val = dask_array.std().compute()
        min_val = dask_array.min().compute()
        max_val = dask_array.max().compute()

    stats = {
        "variable": variable_name,
        "mean": float(mean_val),
        "std_dev": float(std_val),
        "min": float(min_val),
        "max": float(max_val),
    }

    logger.info(f"Statistics for '{variable_name}':")
    for key, value in stats.items():
        logger.info(f"  {key.capitalize()}: {value}")

    if output_path:
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
        description="Compute statistics of a Zarr dataset."
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="/media/jhong90/storage/surya/gong_halpha_2015_2025.zarr",
        help="Path to the Zarr dataset.",
    )
    parser.add_argument(
        "--variable_name",
        type=str,
        default="images",
        help="Name of the variable to compute statistics on.",
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
        default="./data/halpha_stat_train.yaml",
        help="Path to save the computed statistics in YAML format.",
    )
    args = parser.parse_args()

    compute_statistics(
        args.zarr_path, args.variable_name, args.index_path, args.output_path
    )
