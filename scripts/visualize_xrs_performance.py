import os
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
import xarray as xr
from loguru import logger as lgr_logger
from matplotlib import pyplot as plt
import matplotlib as mpl
from omegaconf import DictConfig, OmegaConf

from flare_surya.datamodule import SolarPretrainDataModule
from flare_surya.models import PretrainSolarModel


def setup_plot_style():
    """Set professional plot style for research papers."""
    plt.style.use("seaborn-v0_8-paper")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.usetex": False,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


setup_plot_style()


def visualize_xrs_predictions(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_dir: str,
    num_samples: int = 4,
) -> None:
    """Visualize XRS time series: target, prediction, and delta."""
    batch_size, seq_len = targets.shape
    num_samples = min(num_samples, batch_size)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    time_axis = np.arange(seq_len)

    for i in range(num_samples):
        target = targets[i]
        pred = predictions[i]
        delta = target - pred

        ax_target = axes[i, 0]
        ax_pred = axes[i, 1]
        ax_delta = axes[i, 2]

        ax_target.plot(time_axis, target, label="Target", color="blue", alpha=0.8)
        ax_pred.plot(time_axis, pred, label="Prediction", color="green", alpha=0.8)
        ax_delta.plot(
            time_axis, delta, label="Delta (Target - Pred)", color="red", alpha=0.8
        )
        ax_delta.axhline(y=0, color="black", linestyle="--", linewidth=0.8)

        ax_target.set_title(f"Sample {i} - Target")
        ax_pred.set_title(f"Sample {i} - Prediction")
        ax_delta.set_title(f"Sample {i} - Delta")

        ax_target.set_xlabel("Time step")
        ax_pred.set_xlabel("Time step")
        ax_delta.set_xlabel("Time step")

        ax_target.legend(loc="upper right", fontsize=8)
        ax_pred.legend(loc="upper right", fontsize=8)
        ax_delta.legend(loc="upper right", fontsize=8)

        ax_target.grid(True, alpha=0.3)
        ax_pred.grid(True, alpha=0.3)
        ax_delta.grid(True, alpha=0.3)

    plt.tight_layout(pad=1.5)
    output_path = os.path.join(output_dir, "xrs_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    lgr_logger.info(f"Saved visualization to {output_path}")
    plt.close()


def inverse_norm_log_zscore(data_arr, stats, eps=1e-10):
    """
    Inverse normalize data using z-score and log10.

    Args:
        data_arr: Numpy array.
        stats: DictConfig with 'mean' and 'std'.

    Returns:
        Denormalized data.
    """
    # Denormalize z-score
    x_log = (data_arr * (stats.std + eps)) + stats.mean

    # Denormalize log10
    return 10**x_log


def compute_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    """Compute simple metrics for model performance."""
    mse = np.mean((targets - predictions) ** 2)
    mae = np.mean(np.abs(targets - predictions))
    rmse = np.sqrt(mse)

    target_std = np.std(targets)
    if target_std > 0:
        nrmse = rmse / target_std
    else:
        nrmse = np.nan

    # Compute R2 score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "R2": r2,
    }


@hydra.main(
    version_base=None,
    config_path="../../configs/pretrain/",
    config_name="xrs",
)
def visualize(cfg: OmegaConf) -> None:
    """Main visualization function."""
    output_dir = cfg.etc.get("out_dir", "visualization_output")
    os.makedirs(output_dir, exist_ok=True)

    lgr_logger.info("Setting up datamodule...")
    datamodule = SolarPretrainDataModule(
        zarr_path=cfg.data.zarr_path,
        train_index_path=cfg.data.train_index_path,
        val_index_path=cfg.data.val_index_path,
        test_index_path=cfg.data.test_index_path,
        channels=list(cfg.data.channels),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        data_type=cfg.data.data_type,
        scalers=cfg.data.get("scalers", None),
    )
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    lgr_logger.info("Building model from checkpoint...")
    checkpoint_dir = cfg.etc.get("ckpt_dir", None)
    checkpoint_file = cfg.etc.get("ckpt_file", None)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

    model = PretrainSolarModel.load_from_checkpoint(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lgr_logger.info(f"Using device: {device}")
    model = model.to(device)

    lgr_logger.info("Running inference on test set...")
    all_targets = []
    all_predictions = []

    # Load scalers for denormalization
    scalers_path = cfg.data.get("scalers", None)
    scalers = OmegaConf.load(scalers_path) if scalers_path else None

    # Assuming first channel for denormalization
    channel = cfg.data.channels[0]
    stats = scalers[channel] if scalers and channel in scalers else None

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= cfg.etc.get("max_batches", 5):
                break

            if len(batch) == 3:
                xrs_seq, target, _ = batch
            elif len(batch) == 2:
                xrs_seq, target = batch
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")

            xrs_seq = xrs_seq.to(device)
            target = target.to(device)

            output = model(xrs_seq, use_mask=False)

            if isinstance(output, torch.Tensor):
                predictions = output
            elif isinstance(output, dict):
                predictions = output.get("logits", output.get("pred", None))
                if predictions is None:
                    predictions = list(output.values())[0]
            else:
                predictions = output[0] if isinstance(output, (list, tuple)) else output

            # Squeeze to (B, seq_len)
            if predictions.dim() == 3:
                predictions = predictions.squeeze(1)
            if target.dim() == 3:
                target = target.squeeze(1)

            all_targets.append(target.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    targets_np = np.concatenate(all_targets, axis=0)
    predictions_np = np.concatenate(all_predictions, axis=0)

    # Denormalize if scalers are available
    if stats:
        lgr_logger.info("Denormalizing data for visualization...")
        targets_viz = inverse_norm_log_zscore(targets_np, stats)
        predictions_viz = inverse_norm_log_zscore(predictions_np, stats)
    else:
        targets_viz = targets_np
        predictions_viz = predictions_np

    lgr_logger.info(f"Targets shape: {targets_np.shape}")
    lgr_logger.info(f"Predictions shape: {predictions_np.shape}")

    metrics = compute_metrics(targets_np, predictions_np)
    lgr_logger.info("Metrics:")
    for k, v in metrics.items():
        lgr_logger.info(f"  {k}: {v:.6f}")

    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")
    lgr_logger.info(f"Saved metrics to {metrics_file}")

    visualize_xrs_predictions(
        targets=targets_viz,
        predictions=predictions_viz,
        output_dir=output_dir,
        num_samples=cfg.etc.get("num_viz_samples", 4),
    )

    lgr_logger.info(f"Visualization complete! Output saved to {output_dir}")


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", "Importing from timm.models.layers is deprecated.*", FutureWarning
    )
    visualize()
