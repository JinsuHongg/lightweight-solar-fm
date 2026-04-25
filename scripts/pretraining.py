import warnings

# Suppress the specific FutureWarning from timm
warnings.filterwarnings(
    "ignore", "Importing from timm.models.layers is deprecated.*", FutureWarning
)

import os

import hydra
import torch
import torch.multiprocessing as mp
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger
from loguru import logger as lgr_logger
from omegaconf import OmegaConf

from flare_surya.datamodule import SolarPretrainDataModule
from flare_surya.models import PretrainSolarModel
from flare_surya.utils.callbacks import build_pretrain_callbacks
from flare_surya.utils.logger_utils import build_wandb

torch.set_float32_matmul_precision("medium")


def build_model(cfg):
    """Build the PretrainSolarModel."""
    model = PretrainSolarModel(
        in_channels=cfg.model.in_channels,
        seq_len=cfg.model.seq_len,
        embed_dim=cfg.model.embed_dim,
        encoder_depth=cfg.model.encoder_depth,
        decoder_depth=cfg.model.decoder_depth,
        num_heads=cfg.model.num_heads,
        data_type=cfg.model.data_type,
        save_embeddings_path=cfg.model.get("save_embeddings_path", None),
        mask_ratio=cfg.model.get("mask_ratio", 0.5),
        image_size=cfg.model.get("image_size", 224),
        patch_size=cfg.model.get("patch_size", 16),
        optimizer_dict=cfg.optimizer,
        loss_dict=cfg.loss,
    )
    return model


@hydra.main(
    version_base=None,
    config_path="../../configs/pretrain/",
    config_name="solar_pretrain",
)
def train(cfg: OmegaConf):
    # Datamodule
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

    # Load model
    model = build_model(cfg=cfg)

    # Create wandb logger
    wandb_logger = build_wandb(cfg=cfg)

    # Create CSV logger
    csv_logger = CSVLogger(
        save_dir=os.path.join(cfg.etc.get("log_dir", "logs"), "csv"),
        name=cfg.etc.get("ckpt_name_tag", "pretrain"),
        version=cfg.etc.get("csv_version", None),
    )

    # Trainer
    trainer = pl.Trainer(
        enable_progress_bar=False,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.get("num_nodes", 1),
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        logger=[wandb_logger, csv_logger],
        log_every_n_steps=cfg.logger.get("log_every_n_steps", 50),
        limit_train_batches=cfg.trainer.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.trainer.get("limit_val_batches", 1.0),
        strategy=cfg.trainer.get("strategy", "auto"),
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 1),
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 1.0),
        gradient_clip_algorithm=cfg.trainer.get("gradient_clip_algorithm", "norm"),
        callbacks=build_pretrain_callbacks(cfg),
    )

    lgr_logger.info("Start training...")
    phase = cfg.trainer.get("phase", "train")

    if phase == "train":
        trainer.fit(
            model=model,
            datamodule=datamodule,
        )
    elif phase == "test":
        trainer.test(
            model=model,
            dataloaders=datamodule,
            verbose=True,
        )
    elif phase == "predict":
        # Predict step to save embeddings
        trainer.predict(
            model=model,
            dataloaders=datamodule,
        )


if __name__ == "__main__":
    # Set the start method to 'spawn' for cleaner, safer worker processes.
    # This must be done inside the __main__ block and before any other
    # multiprocessing or CUDA code is called.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Can only be set once

    train()
