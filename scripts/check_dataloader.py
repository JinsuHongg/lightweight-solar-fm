import sys
import os
import torch
import time
import traceback
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Add the project root to the Python path
# This allows us to import from flare_surya
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Imports from your project
from lightweight-solar-fm.dataset import SolarFlareClsDataset
from terratorch_surya.utils.data import build_scalers


def check_dataloader_workers(config_path, num_workers):
    """
    Initializes the dataset and tries to fetch a single batch from a DataLoader
    to diagnose hanging issues with multiprocessing.
    """
    print(f"\n--- Running check with num_workers={num_workers} ---")

    try:
        cfg = OmegaConf.load(config_path)

        # --- Dataset setup from our previous test script ---
        print("Loading scalers...")
        scalers_path = os.path.join(project_root, cfg.data.scalers_path.lstrip("./"))
        scaler_cfg = OmegaConf.load(scalers_path)
        scalers = build_scalers(info=OmegaConf.to_container(scaler_cfg, resolve=True))
        print("Scalers loaded.")

        dataset_args = {
            "sdo_data_root_path": cfg.data.sdo_data_root_path,
            "index_path": cfg.data.train_data_path,
            "flare_index_path": cfg.data.train_flare_data_path,
            "time_delta_input_minutes": list(cfg.data.time_delta_input_minutes),
            "time_delta_target_minutes": cfg.data.time_delta_target_minutes,
            "n_input_timestamps": cfg.data.n_input_timestamps,
            "rollout_steps": cfg.rollout_steps,
            "scalers": scalers,
            "num_mask_aia_channels": cfg.num_mask_aia_channels,
            "drop_hmi_probability": cfg.drop_hmi_probability,
            "use_latitude_in_learned_flow": cfg.use_latitude_in_learned_flow,
            "channels": list(cfg.data.channels),
            "phase": "train",
            "pooling": cfg.data.pooling,
            "random_vert_flip": cfg.data.random_vert_flip,
        }

        print("Initializing dataset...")
        dataset = SolarFlareClsDataset(**dataset_args)
        print(f"Dataset initialized with {len(dataset)} samples.")

        # --- DataLoader test ---
        # We simplify by removing persistent_workers and prefetch_factor,
        # which are only relevant when num_workers > 0 anyway.
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=num_workers,
            shuffle=True,  # Shuffle to avoid hitting the same potentially bad data file
        )

        print(f"Attempting to fetch the first batch...")
        start_time = time.time()

        # This is the line that will hang if there is an issue
        first_batch = next(iter(dataloader))

        end_time = time.time()

        print("\nSuccessfully fetched the first batch!")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        data_dict, _ = first_batch
        print("Data keys and tensor shapes:")
        for key, value in data_dict.items():
            if hasattr(value, "shape"):
                print(f"  - {key}: {value.shape}")

    except Exception as e:
        print(f"\n!!! An error occurred during the test with {num_workers} workers !!!")
        print(f"Error: {e}")
        print("\n--- Full Traceback ---")
        traceback.print_exc()
        print("----------------------")


if __name__ == "__main__":
    config_file = os.path.join(project_root, "configs", "nas", "exp_surya.yaml")

    # First, confirm it works with 0 workers
    check_dataloader_workers(config_file, num_workers=0)

    # Then, test with > 0 workers, which is where it was hanging
    # If this test hangs, we've confirmed the issue is in the
    # dataset loading when multiprocessing is enabled.
    check_dataloader_workers(config_file, num_workers=2)

    print("\n--- Diagnostic script finished ---")
