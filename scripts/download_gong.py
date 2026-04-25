"""
Downloads GONG H-alpha images from the Helioviewer API, resizes them to 512x512,
and stores them in a Zarr v2 file.

Usage:
    python scripts/data/download_gong.py

Dependencies:
    pip install aiohttp backoff hydra-core omegaconf numpy zarr==2.* Pillow loguru imageio imagecodecs
"""

import asyncio
import io
import os
from datetime import datetime, timedelta

import aiohttp
import backoff
import hydra
import imageio.v3 as iio
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from numcodecs import Blosc
from omegaconf import OmegaConf
from PIL import Image
from loguru import logger as lgr_logger

# Constants
TIMEOUT = aiohttp.ClientTimeout(total=60)
API_BASE_URL = "https://api.helioviewer.org/v2/"

# GONG H-alpha sourceId = 94 (verify via getDataSources API if unsure)
# See: https://api.helioviewer.org/docs/v2/appendix/data_sources.html


def get_session() -> aiohttp.ClientSession:
    """Create an aiohttp session."""
    return aiohttp.ClientSession(timeout=TIMEOUT, connector=aiohttp.TCPConnector(ssl=False))


@backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=5)
async def fetch_image_id(
    session: aiohttp.ClientSession,
    date: datetime,
    source_id: int,
    tolerance_seconds: int = 600,
) -> str | None:
    """
    Query getClosestImage to find the JP2 image id nearest to `date`.

    Applies a configurable tolerance: returns None if the closest image is
    more than `tolerance_seconds` away from the requested date.

    Args:
        session: aiohttp session.
        date: The requested UTC datetime.
        source_id: Helioviewer source ID (10 for GONG H-alpha).

    Returns:
        The image id string, or None if outside tolerance or on error.
    """
    # Helioviewer requires a UTC date string ending with Z
    date_str = date.strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "date": date_str,
        "sourceId": source_id,
    }

    try:
        async with session.get(
            f"{API_BASE_URL}getClosestImage/",
            params=params,
        ) as response:
            if response.status == 200:
                data = await response.json(content_type=None)

                if "id" not in data or "date" not in data:
                    lgr_logger.warning(f"Unexpected response for {date_str}: {data}")
                    return None

                # Check 10-minute tolerance.
                # API returns image date as "YYYY-MM-DD HH:MM:SS" (UTC, no timezone).
                image_date = datetime.strptime(data["date"], "%Y-%m-%d %H:%M:%S")
                delta = abs((image_date - date.replace(tzinfo=None)).total_seconds())
                if delta > tolerance_seconds:
                    lgr_logger.debug(
                        f"Skipping {date_str}: closest image is {delta:.0f}s away "
                        f"(image date: {data['date']}, tolerance: {tolerance_seconds}s)"
                    )
                    return None

                lgr_logger.debug(
                    f"Found image id={data['id']} for {date_str} "
                    f"(image date: {data['date']}, delta: {delta:.0f}s)"
                )
                return str(data["id"])
            else:
                lgr_logger.warning(
                    f"getClosestImage failed with status {response.status} for {date_str}"
                )
                return None
    except Exception as e:
        lgr_logger.error(f"Error fetching image id for {date_str}: {e}")
        raise


async def download_image(
    session: aiohttp.ClientSession,
    image_id: str,
    image_size: int = 512,
) -> np.ndarray | None:
    """
    Download a JP2 image by id via getJP2Image and resize to (image_size, image_size).

    Calling getJP2Image with only the `id` parameter (no jpip/json flags) returns
    the raw JP2 binary directly — no URL construction needed.

    PIL cannot decode JPEG 2000 by default — we use imageio + imagecodecs instead.
    GONG H-alpha images are grayscale; stored as (N, H, W, 1) uint8.

    Args:
        session: aiohttp session.
        image_id: Helioviewer image id from getClosestImage.
        image_size: Target width and height in pixels.

    Returns:
        uint8 numpy array of shape (image_size, image_size, 1), or None if failed.
    """
    try:
        lgr_logger.debug(f"Downloading image id={image_id}")
        async with session.get(
            f"{API_BASE_URL}getJP2Image/",
            params={"id": image_id},
        ) as response:
            if response.status == 200:
                img_data = await response.read()

                # imageio + imagecodecs handles JPEG 2000 correctly
                img_array = iio.imread(io.BytesIO(img_data))

                # Normalise to uint8 if the JP2 is 16-bit
                if img_array.dtype != np.uint8:
                    img_max = img_array.max()
                    if img_max > 0:
                        img_array = img_array.astype(np.float32) / img_max * 255
                    img_array = img_array.clip(0, 255).astype(np.uint8)

                # Resize using PIL (LANCZOS for quality)
                pil_img = Image.fromarray(img_array)
                pil_img = pil_img.resize((image_size, image_size), Image.LANCZOS)
                img_array = np.array(pil_img)

                # GONG H-alpha is single-channel grayscale -> shape (H, W, 1)
                if img_array.ndim == 2:
                    img_array = img_array[:, :, np.newaxis]
                elif img_array.ndim == 3 and img_array.shape[2] != 1:
                    # Unexpected multi-channel: convert to grayscale luminance
                    img_array = np.mean(
                        img_array[:, :, :3], axis=2, keepdims=True
                    ).astype(np.uint8)

                return img_array
            else:
                lgr_logger.warning(
                    f"getJP2Image failed for id={image_id}: HTTP {response.status}"
                )
                return None
    except Exception as e:
        lgr_logger.error(f"Error downloading image id={image_id}: {e}")
        return None


async def process_batch(
    session: aiohttp.ClientSession,
    dates: list[datetime],
    source_id: int,
    semaphore: asyncio.Semaphore,
    image_size: int,
    tolerance_seconds: int = 600,
) -> list[tuple[datetime, np.ndarray | None]]:
    """
    Process a batch of dates concurrently, deduplicating by image id.

    Multiple requested dates often resolve to the same underlying image (because
    GONG's cadence is ~15-20 min but the requested cadence may be finer). We fetch
    all ids first, then download each unique id only once, and reuse the result for
    every date that mapped to it.

    Args:
        session: aiohttp session.
        dates: List of dates to process.
        source_id: Data source ID.
        semaphore: Concurrency semaphore.
        image_size: Target image size.

    Returns:
        List of (date, image_array) tuples. Dates whose closest image is outside
        the 10-minute tolerance, or that failed to download, have None as the array.
    """

    # --- Step 1: fetch all image ids concurrently ---
    async def _fetch_id(date: datetime) -> tuple[datetime, str | None]:
        async with semaphore:
            return date, await fetch_image_id(
                session, date, source_id, tolerance_seconds
            )

    id_results: list[tuple[datetime, str | None]] = await asyncio.gather(
        *[_fetch_id(d) for d in dates]
    )

    # --- Step 2: download each unique id exactly once ---
    unique_ids: set[str] = {img_id for _, img_id in id_results if img_id is not None}
    if unique_ids:
        lgr_logger.debug(
            f"Batch: {len(dates)} dates -> {len(unique_ids)} unique images "
            f"({len(dates) - len(unique_ids)} duplicates skipped)"
        )

    async def _download_unique(img_id: str) -> tuple[str, np.ndarray | None]:
        async with semaphore:
            return img_id, await download_image(session, img_id, image_size)

    downloaded: dict[str, np.ndarray | None] = dict(
        await asyncio.gather(*[_download_unique(uid) for uid in unique_ids])
    )

    # --- Step 3: map results back to each requested date ---
    return [
        (date, downloaded.get(img_id) if img_id is not None else None)
        for date, img_id in id_results
    ]


@hydra.main(
    version_base=None,
    config_path="../../configs/data/",
    config_name="gong_download",
)
def main(cfg: OmegaConf) -> None:
    """Main entry point."""
    lgr_logger.info("Starting GONG H-alpha data download...")

    start_date = datetime.fromisoformat(cfg.download.start_date.replace("Z", "+00:00"))
    end_date = datetime.fromisoformat(cfg.download.end_date.replace("Z", "+00:00"))
    cadence = timedelta(minutes=cfg.download.cadence_minutes)
    image_size: int = cfg.download.image_size
    tolerance_seconds: int = cfg.download.tolerance_seconds

    dates: list[datetime] = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += cadence

    lgr_logger.info(f"Total dates to process: {len(dates)}")

    output_dir = os.path.join(cfg.output.output_dir, cfg.output.zarr_name)
    os.makedirs(output_dir, exist_ok=True)

    total_saved = 0

    async def run_download() -> None:
        nonlocal total_saved
        is_first_write = True
        semaphore = asyncio.Semaphore(cfg.download.max_concurrent)
        batch_size = 100
        num_batches = (len(dates) + batch_size - 1) // batch_size

        async with get_session() as session:
            for batch_idx, i in enumerate(range(0, len(dates), batch_size)):
                batch_dates = dates[i : i + batch_size]
                lgr_logger.info(
                    f"Processing batch {batch_idx + 1} / {num_batches} "
                    f"({len(batch_dates)} dates)..."
                )

                results = await process_batch(
                    session,
                    batch_dates,
                    cfg.data_source_id,
                    semaphore,
                    image_size,
                    tolerance_seconds,
                )

                batch_imgs = [img for _, img in results if img is not None]
                batch_ts = [
                    pd.to_datetime(date).tz_localize(None) for date, img in results if img is not None
                ]

                if batch_imgs:
                    imgs_array = np.stack(batch_imgs, axis=0)  # (B, H, W, 1)

                    ds_batch = xr.Dataset(
                        {
                            "images": (["timestep", "y", "x", "channel"], imgs_array),
                        },
                        coords={
                            "timestep": pd.to_datetime(batch_ts),
                            "y": np.arange(image_size),
                            "x": np.arange(image_size),
                            "channel": ["halpha"],
                        },
                    )

                    encoding = {
                        "images": {
                            "compressor": Blosc(
                                cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE
                            ),
                            "chunks": (100, image_size, image_size, 1),
                        },
                        "timestep": {
                            "dtype": "float64",
                            "units": "hours since 2010-04-08 00:00:00",
                            "calendar": "proleptic_gregorian",
                        },
                    }

                    if is_first_write:
                        if os.path.exists(output_dir):
                            import shutil
                            shutil.rmtree(output_dir)
                        os.makedirs(output_dir, exist_ok=True)

                        ds_batch.to_zarr(output_dir, mode="w", encoding=encoding)
                        is_first_write = False
                    else:
                        ds_batch.to_zarr(output_dir, append_dim="timestep")

                    total_saved += len(batch_imgs)
                    lgr_logger.info(
                        f"Batch {batch_idx + 1}: saved {len(batch_imgs)} images "
                        f"(total so far: {total_saved})"
                    )

        lgr_logger.info(
            f"Done. Saved {total_saved} images out of {len(dates)} requested "
            f"to {output_dir}"
        )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_download())
    finally:
        loop.close()

    if total_saved == 0:
        lgr_logger.warning("No images were downloaded.")
    else:
        zarr.consolidate_metadata(output_dir)
        lgr_logger.info(f"Consolidated zarr store at {output_dir}")


if __name__ == "__main__":
    main()
