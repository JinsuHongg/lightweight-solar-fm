import numpy as np
import pandas as pd
import torch
import xarray as xr
from loguru import logger as lgr_logger
from omegaconf import DictConfig
from torch.utils.data import Dataset


class SolarPretrainDataset(Dataset):
    """
    A dataset for self-supervised pre-training of solar data using Zarr.

    This dataset loads raw solar data (either 1D time-series or 2D images)
    from Zarr and returns them as (input, target) pairs for reconstruction training.

    It expects:
    - A Zarr store containing the data.
    - An index file (CSV) containing timestamps for the split.
    """

    def __init__(
        self,
        zarr_path: str,
        index_path: str,
        channels: list[str],
        scalers: DictConfig | None = None,
        data_type: str = "1d",
        phase: str = "train",
        transform=None,
    ):
        """
        Args:
            zarr_path (str): Path to the Zarr store.
            index_path (str): Path to the index CSV file (containing timestamps).
            channels (list[str]): List of channels to load from Zarr.
            scalers (DictConfig, optional): Normalization statistics.
            data_type (str): Type of data, either '1d' or '2d'.
            phase (str): Phase of the dataset (train, val, test).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.zarr_path = zarr_path
        self.index_path = index_path
        self.channels = channels
        self.scalers = scalers
        self.data_type = data_type
        self.phase = phase
        self.transform = transform

        # Load index
        lgr_logger.info(f"Loading index from {index_path}")
        self.index = pd.read_csv(index_path)
        self.index["timestamp"] = pd.to_datetime(self.index["timestamp"]).values.astype(
            "datetime64[ns]"
        )
        self.index.set_index("timestamp", inplace=True)
        self.index.sort_index(inplace=True)

        # Don't open Zarr store here - each worker will open its own lazily in __getitem__
        lgr_logger.info(f"Zarr store path stored: {zarr_path}")
        self._zarr_data = None

        # Validate index against available Zarr timestamps
        # Handle timestamp misalignment: filter timestamps from index that don't exist in Zarr
        # We only check this if we're in the main process (len(index) is reasonable)
        # In worker processes, we'll skip this check to avoid opening Zarr for each worker
        if self._zarr_data is None:
            self._open_zarr()

        # Convert index timestamps to string format for matching
        # Use 1-minute tolerance to handle precision issues
        index_str = self.index.index.strftime("%Y-%m-%d %H:%M")

        # Build Zarr timestamp strings with same precision (hour:minute) for matching
        available_timestamps_str = set()
        for var_name in self._zarr_data.data_vars:
            var_data = self._zarr_data[var_name]
            if "timestep" in var_data.dims:
                zarr_times = var_data.timestep.values
                for t in zarr_times:
                    # cftime objects have strftime method
                    if hasattr(t, "strftime"):
                        available_timestamps_str.add(t.strftime("%Y-%m-%d %H:%M"))
                    else:
                        # Handle float timestamps if not decoded
                        import cftime as cf

                        decoded = cf.num2date(
                            t,
                            units=var_data.timestep.attrs.get(
                                "units", "hours since 2010-04-08 00:00:00"
                            ),
                            calendar=var_data.timestep.attrs.get(
                                "calendar", "proleptic_gregorian"
                            ),
                        )
                        available_timestamps_str.add(decoded.strftime("%Y-%m-%d %H:%M"))

        original_length = len(self.index)
        self.index = self.index[index_str.isin(available_timestamps_str)]
        filtered_length = len(self.index)

        if original_length != filtered_length:
            lgr_logger.info(
                f"Filtered {original_length - filtered_length} samples from index "
                f"(not found in Zarr data). "
                f"Using {filtered_length} samples."
            )

        self.length = len(self.index)

    def _open_zarr(self):
        """Open Zarr store lazily. Each worker opens its own handle."""
        if self._zarr_data is None:
            lgr_logger.info(f"Opening Zarr store at {self.zarr_path}")
            # Use CFDatetimeCoder for proper time decoding
            from xarray.coding.times import CFDatetimeCoder

            time_coder = CFDatetimeCoder(use_cftime=True)
            self._zarr_data = xr.open_zarr(
                self.zarr_path,
                consolidated=True,
                decode_times=time_coder,
            )

    def __len__(self):
        return self.length

    def norm_log_zscore(self, data_arr, stats, eps=1e-10):
        """
        Normalize data using log10 and z-score.

        Args:
            data_arr: Numpy array or Xarray DataArray.
            stats: DictConfig with 'mean' and 'std'.

        Returns:
            Normalized data.
        """
        x = np.clip(data_arr, eps, None)  # avoid log(0)
        x_log = np.log10(x)

        # Add epsilon to std dev to prevent division by zero
        std_dev = stats.std + eps
        result = (x_log - stats.mean) / std_dev

        if np.any(np.isnan(result)):
            lgr_logger.warning("NaNs detected in normalization output.")
            lgr_logger.warning(
                f"Input data min/max: {np.min(data_arr)}, {np.max(data_arr)}"
            )
            lgr_logger.warning(f"Stats: mean={stats.mean}, std={stats.std}")
            nan_mask = np.isnan(result)
            lgr_logger.warning(
                f"Original data values at NaN locations: {data_arr[nan_mask]}"
            )
            lgr_logger.warning(f"log10 values at NaN locations: {x_log[nan_mask]}")

        return result

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (input, target, timestamp) where both are tensors.
        """
        # Open Zarr lazily if not already open
        if self._zarr_data is None:
            self._open_zarr()

        # Get timestamp (stored as string) and convert back for Zarr selection
        timestamp_str = self.index.index[idx]

        # Convert string back to cftime for selection (to match decoded Zarr data)
        timestamp_dt = pd.to_datetime(timestamp_str)
        # Use cftime for exact matching (same precision as Zarr storage)
        import cftime

        calendar = self._zarr_data.timestep.encoding.get(
            "calendar", "proleptic_gregorian"
        )
        timestamp_cftime = cftime.datetime(
            timestamp_dt.year,
            timestamp_dt.month,
            timestamp_dt.day,
            timestamp_dt.hour,
            timestamp_dt.minute,
            0,  # always 0 seconds for hour-aligned data
            calendar=calendar,
        )

        # Timestamp string for easy return
        timestamp = timestamp_str

        # Detect data format:
        # Format A: Single variable with channel dimension (e.g., 'xray' with dims: timestep, minute_offset, channel)
        # Format B: Separate variables per channel (e.g., 'soft', 'hard' as separate data vars)

        channel_data = []
        first_var = list(self._zarr_data.data_vars.keys())[0]
        first_var_dims = self._zarr_data[first_var].dims

        if "channel" in first_var_dims:
            # Format A: Single variable with channel dimension
            try:
                da = self._zarr_data[first_var].sel(timestep=timestamp_cftime)
            except KeyError:
                lgr_logger.error(
                    f"Timestamp {timestamp_cftime} not found in Zarr data variable {first_var}."
                )
                raise IndexError(f"Timestamp {timestamp_cftime} not found in Zarr.")

            # da now has shape (minute_offset, channel)
            # Extract each channel and stack them
            for ch in self.channels:
                try:
                    ch_data = da.sel(channel=ch).values
                except KeyError:
                    lgr_logger.error(f"Channel {ch} not found in Zarr.")
                    raise IndexError(f"Channel {ch} not found in Zarr.")

                # Normalize if scaler is available
                if self.scalers and ch in self.scalers:
                    stats = self.scalers[ch]
                    ch_data = self.norm_log_zscore(ch_data, stats)

                channel_data.append(ch_data)
        else:
            # Format B: Separate variables per channel
            for ch in self.channels:
                try:
                    da = self._zarr_data[ch].sel(timestep=timestamp_cftime)
                except KeyError:
                    lgr_logger.error(
                        f"Channel {ch} at timestamp {timestamp_cftime} not found in Zarr."
                    )
                    raise IndexError(
                        f"Channel {ch} at timestamp {timestamp_cftime} not found in Zarr."
                    )

                # Get numpy array from DataArray
                data_np = np.array(da.values)

                # Normalize if scaler is available
                if self.scalers and ch in self.scalers:
                    stats = self.scalers[ch]
                    data_np = self.norm_log_zscore(data_np, stats)

                channel_data.append(data_np)

        # Stack channels together: (channels, minute_offset)
        data_np = np.stack(channel_data, axis=0)

        # Convert to tensor
        data_tensor = torch.tensor(data_np, dtype=torch.float32)

        # Handle transform
        if self.transform:
            data_tensor = self.transform(data_tensor)

        # For self-supervised, input and target are the same
        # Convert timestamp to string for PyTorch DataLoader collation
        timestamp_str = str(timestamp)
        return data_tensor, data_tensor, timestamp_str
