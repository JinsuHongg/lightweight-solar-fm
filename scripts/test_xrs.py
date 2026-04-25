import os
import torch
import hydra
import numpy as np
from loguru import logger as lgr_logger
from omegaconf import OmegaConf

from flare_surya.datamodule import SolarPretrainDataModule
from flare_surya.models import PretrainSolarModel


def compute_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    """Compute performance metrics."""
    mse = np.mean((targets - predictions) ** 2)
    mae = np.mean(np.abs(targets - predictions))
    rmse = np.sqrt(mse)

    target_std = np.std(targets)
    nrmse = rmse / target_std if target_std > 0 else np.nan

    # R2 score
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
def test(cfg: OmegaConf) -> None:
    """Evaluate model on the entire test set."""
    output_dir = cfg.etc.get("save_test_results_path", "./results/test/")
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

    lgr_logger.info("Loading model from checkpoint...")
    checkpoint_path = os.path.join(cfg.etc.ckpt_dir, cfg.etc.ckpt_file)
    model = PretrainSolarModel.load_from_checkpoint(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    lgr_logger.info(f"Using device: {device}")

    lgr_logger.info("Running inference on entire test set...")
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
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

            if predictions.dim() == 3:
                predictions = predictions.squeeze(1)
            if target.dim() == 3:
                target = target.squeeze(1)

            all_targets.append(target.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    targets_np = np.concatenate(all_targets, axis=0)
    predictions_np = np.concatenate(all_predictions, axis=0)

    metrics = compute_metrics(targets_np, predictions_np)

    lgr_logger.info("Final Test Metrics:")
    for k, v in metrics.items():
        lgr_logger.info(f"  {k}: {v:.6f}")

    metrics_file = os.path.join(output_dir, "test_metrics.txt")
    with open(metrics_file, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")
    lgr_logger.info(f"Saved metrics to {metrics_file}")


if __name__ == "__main__":
    test()
