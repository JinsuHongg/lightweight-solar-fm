# lw-solar-fm

## Introduction

This repository hosts a lightweight, domain-specific foundation model for solar physics data, focusing on capturing high-cadence temporal dynamics. This model is designed for self-supervised pre-training on GOES X-ray flux (XRS) timeseries data using Masked Time Series Modeling (MTM). The goal is to create generalized temporal embeddings that can be seamlessly integrated with large-scale spatial foundation models for comprehensive multimodal forecasting.

## Key Features

- **Lightweight Foundation Model:** Optimized for efficiency and domain specificity.
- **Self-Supervised Learning (SSL):** Pre-training using Masked Time Series Modeling (MTM) on GOES XRS data.
- **Temporal Dynamics Extraction:** Captures critical high-cadence temporal context for space weather forecasting.
- **Scalable Architecture:** Designed for seamless integration with spatial foundation models (e.g., Surya).
- **Plug-and-Play Embeddings:** Provides generalized temporal representations for multimodal forecasting.

## Installation

This repository can be installed using Conda/Mamba. Please ensure you have Conda or Mamba installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/lw-solar-fm.git
    cd lw-solar-fm
    ```

2.  **Create a Conda environment:**
    Use the provided `environment.yml` file to create a new environment:
    ```bash
    conda env create -f environment.yml
    ```
    This command will create a new environment named `lw-solar-fm` with all the necessary dependencies.

3.  **Activate the environment:**
    ```bash
    conda activate lw-solar-fm
    ```

4.  **Install the package in editable mode:**
    This allows you to make changes to the package code and have them immediately reflected without needing to reinstall.
    ```bash
    pip install -e .
    ```

## Data Preparation (Zarr Files)

This section details how to prepare your solar data into Zarr files, which are used by the datamodules.

### GOES X-ray Flux (XRS) Data Preparation

The repository includes a script, `scripts/preprocessing_xrs.py`, to process raw GOES XRS NetCDF files and convert them into a Zarr file. This script is crucial for preparing the data used in the pre-training tasks.

1.  **Download GOES XRS Data:**
    You can download GOES XRS Level 2 Average 1-minute cadence data from the NOAA data portal:
    *   **GOES 16/17/18:** Access data via `https://www.ncei.noaa.gov/products/goes-r-extreme-ultraviolet-xray-irradiance`
    *   **GOES 15:** Access data via `https://www.ncei.noaa.gov/products/goes-1-15/space-weather-instruments`
    Ensure you download the data in NetCDF (`.nc`) format.

2.  **Configure the preprocessing script:**
    The `scripts/preprocessing_xrs.py` script uses Python's default arguments. You may need to modify the `if __name__ == "__main__":` block within the script to set:
    *   `data_dir`: The directory where you have downloaded the `.nc` files.
    *   `zarr_target`: The desired output path for the Zarr file (e.g., `./data/xrs_24hour_slices_v2.zarr`).
    *   `window_size`, `step_size`: The window size (in hours) and step size (in hours) for creating time series segments.
    *   `time_filters`: Optional dictionary to filter data by satellite and date range.

3.  **Run the preprocessing script:**
    After configuring the paths and parameters in the script, navigate to the `scripts/` directory and run it:
    ```bash
    cd scripts
    python preprocessing_xrs.py
    ```
    This will process the specified `.nc` files and generate the Zarr file at the designated `zarr_target` path.

*Note: Ensure you have installed all necessary dependencies, including `xarray`, `zarr`, `numpy`, `pandas`, `loguru`, `numcodecs`, and `imageio`. These are typically included in the `environment.yml`.*

### GONG H-alpha Data

The repository includes a script to download GONG H-alpha images and convert them into a Zarr file.

1.  **Install dependencies:**
    Ensure you have the necessary libraries installed. The `environment.yml` file includes most of them. You might need to install `imageio` and `imagecodecs` separately if they are not already included:
    ```bash
    pip install imageio imagecodecs
    ```

2.  **Configure the download script:**
    You will need to create or modify a Hydra configuration file for the download script. Look for a configuration file related to data download (e.g., `configs/data/gong_download.yaml`) and set the following parameters:
    *   `download.start_date`: The starting date for the data download (e.g., `2023-01-01T00:00:00Z`).
    *   `download.end_date`: The ending date for the data download (e.g., `2023-01-02T00:00:00Z`).
    *   `download.cadence_minutes`: The desired time cadence in minutes (e.g., `15`).
    *   `download.image_size`: The target image resolution (e.g., `512`).
    *   `download.tolerance_seconds`: Tolerance for matching image timestamps (e.g., `600`).
    *   `download.max_concurrent`: Maximum concurrent downloads (e.g., `10`).
    *   `output.output_dir`: The directory where the Zarr store will be saved.
    *   `output.zarr_name`: The name for the output Zarr file (e.g., `gong_halpha.zarr`).
    *   `data_source_id`: The Helioviewer source ID for GONG H-alpha (typically `94` or `10`, verify via API if unsure).

3.  **Run the download script:**
    Navigate to the `scripts/` directory and execute the `download_gong.py` script, specifying your configuration file:
    ```bash
    cd scripts
    python download_gong.py --config-path=../../configs/data --config-name=gong_download
    ```
    *Note: Adjust the `--config-path` and `--config-name` as needed based on where you save your configuration file.*

## Running Pretraining

To run the pre-training script, you will use Hydra to manage configurations.

1.  **Navigate to the scripts directory:**
    ```bash
    cd scripts
    ```

2.  **Execute the pretraining script:**
    You need to specify the configuration file to use. If your configuration file is named `solar_pretrain.yaml` and is located in `configs/pretrain/`, you can run it as follows:
    ```bash
    python pretraining.py --config-name=solar_pretrain
    ```
    If your configuration file is `xrs.yaml` and is located in `configs/`, and you have modified `pretraining.py` to point to it (e.g., `config_name="xrs"`), you would run:
    ```bash
    python pretraining.py --config-name=xrs
    ```
    *Note: Adjust the `--config-name` and potentially the `config_path` in the `pretraining.py` script if your configuration files are organized differently.*

## Running Test Script (test_xrs.py)

The `test_xrs.py` script evaluates the trained model on the test set and computes performance metrics.

1.  **Navigate to the scripts directory:**
    ```bash
    cd scripts
    ```

2.  **Execute the test script:**
    This script also uses Hydra for configuration. It expects a configuration file (e.g., `xrs.yaml` as defined in the script's `@hydra.main` decorator).
    ```bash
    python test_xrs.py --config-name=xrs
    ```
    *Note: Ensure that the `ckpt_dir` and `ckpt_file` within your specified configuration file point to a valid trained model checkpoint.*

## Usage


## Contributing


## License

