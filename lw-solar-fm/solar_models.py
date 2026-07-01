import torch
import torch.nn as nn


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # The Shortcut path (identity or projection)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Addition step
        out += residual
        return self.act(out)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # The Shortcut path (identity or projection)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Addition step
        out += residual
        return self.act(out)


class SolarTokenizer1D(nn.Module):
    def __init__(self, in_channels=2, embed_dim=768):
        super().__init__()
        self.layer1 = ResidualBlock1D(in_channels, embed_dim, kernel_size=7)
        self.layer2 = ResidualBlock1D(embed_dim, embed_dim, kernel_size=5)

    def forward(self, x):
        # x: [Batch, 1, seq_len]
        x = self.layer1(x)
        x = self.layer2(x)

        # Prepare for Transformer [Batch, Seq_Len, Dim]
        return x.transpose(1, 2)


class SolarTokenizer2D(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, image_size=224, patch_size=16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SolarViTBlock1D(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # Pre-normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Multi-Head Self Attention
        # batch_first=True ensures we use [Batch, Seq_Len, Dim]
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # The MLP (Feed-Forward) block with 4x expansion
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),  # Standard activation for ViT architectures
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x shape: [Batch, seq_len, embed_dim]

        # 1. Attention path with Residual
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output

        # 2. MLP path with Residual
        x = x + self.mlp(self.norm2(x))

        return x


class SolarSequenceEncoder(nn.Module):
    def __init__(self, seq_len=1440, embed_dim=768, depth=4, num_heads=12):
        super().__init__()

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        # Standard ViT practice is to drop after adding pos_embed
        self.pos_drop = nn.Dropout(p=0.1)

        # Stack the ViT blocks
        self.blocks = nn.ModuleList(
            [
                SolarViTBlock1D(embed_dim=embed_dim, num_heads=num_heads)
                for _ in range(depth)
            ]
        )

        # Final norm (standard in ViT before pooling or fusion)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        # Truncated normal initialization for the positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x is the output from SecondaryTokenizer: [Batch, seq_len, embed_dim]

        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through all Transformer blocks
        for block in self.blocks:
            x = block(x)

        return self.norm(x)


class SolarEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        seq_len=1440,
        embed_dim=768,
        depth=4,
        num_heads=12,
        data_type="1d",
        image_size=224,
        patch_size=16,
    ):
        super().__init__()
        self.data_type = data_type

        if data_type == "1d":
            self.tokenizer = SolarTokenizer1D(in_channels, embed_dim)
            # For 1D, seq_len is the sequence length
            self.encoder = SolarSequenceEncoder(seq_len, embed_dim, depth, num_heads)
        elif data_type == "2d":
            self.tokenizer = SolarTokenizer2D(
                in_channels, embed_dim, image_size, patch_size
            )
            # For 2D, seq_len is the number of patches
            num_patches = (image_size // patch_size) ** 2
            self.encoder = SolarSequenceEncoder(
                num_patches, embed_dim, depth, num_heads
            )
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    def forward(self, x):
        token = self.tokenizer(x)
        embedding = self.encoder(token)

        return embedding


class SolarSequenceDecoder(nn.Module):
    def __init__(self, seq_len=1440, embed_dim=768, depth=2, num_heads=12):
        super().__init__()

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        # Standard ViT practice is to drop after adding pos_embed
        self.pos_drop = nn.Dropout(p=0.1)

        # Stack the ViT blocks
        self.blocks = nn.ModuleList(
            [
                SolarViTBlock1D(embed_dim=embed_dim, num_heads=num_heads)
                for _ in range(depth)
            ]
        )

        # Final norm (standard in ViT before pooling or fusion)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        # Truncated normal initialization for the positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x is the output from Encoder: [Batch, seq_len, embed_dim]

        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through all Transformer blocks
        for block in self.blocks:
            x = block(x)

        return self.norm(x)


class SolarDetokenizer1D(nn.Module):
    def __init__(self, in_channels=2, embed_dim=768):
        super().__init__()
        # Invert the tokenization: embed_dim -> in_channels
        # We use transposed convolutions (ConvTranspose1d) to upsample
        self.layer1 = ResidualBlock1D(embed_dim, embed_dim, kernel_size=5)
        self.layer2 = ResidualBlock1D(embed_dim, in_channels, kernel_size=7)

    def forward(self, x):
        # x: [Batch, seq_len, embed_dim] - Output from SequenceDecoder
        # We need to convert back to [Batch, in_channels, seq_len]

        # First convert to [Batch, embed_dim, seq_len]
        x = x.transpose(1, 2)

        x = self.layer1(x)
        x = self.layer2(x)

        return x


class SolarDetokenizer2D(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, image_size=224):
        super().__init__()
        self.image_size = image_size
        self.embed_dim = embed_dim
        patch_size = 16
        num_patches_per_side = image_size // patch_size
        self.num_patches_per_side = num_patches_per_side

        self.proj = nn.ConvTranspose2d(
            embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.layer1 = ResidualBlock2D(
            embed_dim, embed_dim // 2, kernel_size=3, stride=1
        )
        self.layer2 = ResidualBlock2D(
            embed_dim // 2, in_channels, kernel_size=3, stride=1
        )

    def forward(self, x):
        B, num_patches, D = x.shape
        H = W = self.num_patches_per_side

        x = x.transpose(1, 2).reshape(B, D, H, W)
        x = self.proj(x)
        x = self.layer1(x)
        x = self.layer2(x)

        return x


class SolarDecoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        seq_len=1440,
        embed_dim=768,
        depth=2,
        num_heads=12,
        data_type="1d",
        image_size=224,
    ):
        super().__init__()
        self.data_type = data_type

        self.sequence_decoder = SolarSequenceDecoder(
            seq_len, embed_dim, depth, num_heads
        )

        if data_type == "1d":
            self.detokenizer = SolarDetokenizer1D(in_channels, embed_dim)
        elif data_type == "2d":
            self.detokenizer = SolarDetokenizer2D(in_channels, embed_dim, image_size)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    def forward(self, x):
        x = self.sequence_decoder(x)
        reconstruction = self.detokenizer(x)
        return reconstruction

    def decode_sequence(self, x):
        return self.sequence_decoder(x)


class SolarPretrainingMetrics(MetricCollection):
    def __init__(self, prefix: str = None, postfix: str = None):
        super().__init__(
            {
                "mae": MeanAbsoluteError(),
                "mse": MeanSquaredError(),
                "rmse": MeanSquaredError(squared=False),
                "r2": GlobalR2Score(),
            },
            prefix=prefix,
            postfix=postfix,
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds_flat = preds.reshape(-1)
        target_flat = target.reshape(-1)
        super().update(preds_flat, target_flat)


class PretrainSolarModel(pl.LightningModule):
    def __init__(
        self,
        in_channels=1,
        seq_len=1440,
        embed_dim=768,
        encoder_depth=4,
        decoder_depth=2,
        num_heads=12,
        data_type="1d",
        save_embeddings_path: str | None = None,
        mask_ratio: float = 0.5,
        image_size: int = 224,
        patch_size: int = 16,
        optimizer_dict=None,
        loss_dict=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_type = data_type
        self.image_size = image_size
        self.patch_size = patch_size

        self.optimizer_dict = optimizer_dict or {
            "type": "adamw",
            "lr": 1e-4,
            "weight_decay": 0.01,
            "eps": 1e-8,
            "betas": [0.9, 0.999],
            "scheduler": {
                "use": "cosine_warmup",
                "monitor": "val/loss",
                "cosine_warmup": {
                    "total_steps": 10000,
                    "warmup_ratio": 0.1,
                    "min_lr": 1e-6,
                },
            },
        }
        self.loss_dict = loss_dict or {"type": "mse"}

        self.encoder = SolarEncoder(
            in_channels=in_channels,
            seq_len=seq_len,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            data_type=data_type,
            image_size=image_size,
            patch_size=patch_size,
        )

        self.decoder = SolarDecoder(
            in_channels=in_channels,
            seq_len=seq_len,
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=num_heads,
            data_type=data_type,
            image_size=image_size,
            patch_size=patch_size,
        )

        self.data_type = data_type
        self.save_embeddings_path = save_embeddings_path
        self.mask_ratio = mask_ratio

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.pred_timestamps = []
        self.pred_embeddings = []
        self._last_mask_indices = None
        self._last_seq_mask_indices = None

        self.train_metrics = SolarPretrainingMetrics(prefix="train/")
        self.val_metrics = SolarPretrainingMetrics(prefix="val/")
        self.test_metrics = SolarPretrainingMetrics(prefix="test/")

    def random_mask(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = tokens.shape
        num_mask = int(N * self.mask_ratio)

        noise = torch.rand(B, N, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_mask = ids_shuffle[:, :num_mask]

        tokens_masked = tokens.clone()
        mask_token = self.mask_token.to(tokens.dtype).to(tokens.device)

        mask_2d = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
        mask_2d.scatter_(1, ids_mask, torch.ones_like(ids_mask, dtype=torch.bool))
        tokens_masked = torch.where(mask_2d.unsqueeze(-1), mask_token, tokens_masked)

        return tokens_masked, ids_restore, ids_mask

    def forward(self, x, use_mask: bool = True):
        tokens = self.encoder.tokenizer(x)
        nan_check = torch.isnan(tokens)
        if nan_check.any():
            raise ValueError(
                f"NaN detected in tokenizer output: {nan_check.sum()} / {tokens.numel()}"
            )

        if use_mask and self.training and self.mask_ratio:
            tokens, ids_restore, ids_mask = self.random_mask(tokens)
            self._last_mask_indices = ids_mask
            self._last_seq_mask_indices = ids_mask
        else:
            self._last_mask_indices = None
            self._last_seq_mask_indices = None

        embedding = self.encoder.encoder(tokens)
        nan_check = torch.isnan(embedding)
        if nan_check.any():
            raise ValueError(
                f"NaN detected in encoder output: {nan_check.sum()} / {embedding.numel()}"
            )

        decoded = self.decoder.sequence_decoder(embedding)

        reconstruction = self.decoder.detokenizer(decoded)

        nan_check = torch.isnan(reconstruction)
        if nan_check.any():
            raise ValueError(
                f"NaN detected in detokenizer output: {nan_check.sum()} / {reconstruction.numel()}"
            )

        return reconstruction

    def encode(self, x):
        """Encode input to embeddings without decoding."""
        return self.encoder(x)

    def _compute_loss(self, pred, target):
        loss_type = self.loss_dict.get("type", "mse").lower()
        if loss_type == "mae":
            return nn.functional.l1_loss(pred, target)
        elif loss_type == "rmse":
            return torch.sqrt(nn.functional.mse_loss(pred, target))
        else:
            return nn.functional.mse_loss(pred, target)

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        if torch.isnan(x).any():
            raise ValueError(
                f"NaN detected in input x: {torch.isnan(x).sum()} / {x.numel()}"
            )
        if torch.isnan(y).any():
            raise ValueError(
                f"NaN detected in input y: {torch.isnan(y).sum()} / {y.numel()}"
            )

        pred = self.forward(x, use_mask=True)
        mask_indices = self._last_seq_mask_indices

        if mask_indices is not None:
            if self.data_type == "2d":
                patch_size = self.encoder.tokenizer.patch_size
                image_size = self.encoder.tokenizer.image_size
                num_patches_per_side = image_size // patch_size

                def to_patches(tensor):
                    B, C, H, W = tensor.shape
                    patches = tensor.unfold(2, patch_size, patch_size).unfold(
                        3, patch_size, patch_size
                    )
                    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
                        B,
                        num_patches_per_side * num_patches_per_side,
                        C,
                        patch_size * patch_size,
                    )
                    return patches

                y_patches = to_patches(y)
                pred_patches = to_patches(pred)

                B_mask, num_mask = mask_indices.shape
                y_masked = y_patches[
                    torch.arange(B_mask, device=mask_indices.device).unsqueeze(1),
                    mask_indices,
                ]
                pred_masked = pred_patches[
                    torch.arange(B_mask, device=mask_indices.device).unsqueeze(1),
                    mask_indices,
                ]
            else:
                y_masked = torch.gather(
                    y,
                    dim=2,
                    index=mask_indices.unsqueeze(1).expand(-1, y.shape[1], -1),
                )
                pred_masked = torch.gather(
                    pred,
                    dim=2,
                    index=mask_indices.unsqueeze(1).expand(-1, pred.shape[1], -1),
                )
            loss = self._compute_loss(pred_masked, y_masked)
            self.train_metrics.update(pred_masked, y_masked)
        else:
            loss = self._compute_loss(pred, y)
            self.train_metrics.update(pred, y)

        if torch.isnan(loss):
            raise ValueError(
                f"NaN detected in loss. pred stats: min={pred.min()}, max={pred.max()}, mean={pred.mean()}"
            )

        self.log("train/loss", loss, prog_bar=True, batch_size=x.shape[0], on_step=True)
        return loss

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict({k: v for k, v in metrics.items()})
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        pred = self.forward(x, use_mask=False)

        self.val_metrics.update(pred, y)

        loss = nn.functional.mse_loss(pred, y)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True, batch_size=x.shape[0])

    def on_validation_epoch_end(self):

        metrics = self.val_metrics.compute()
        self.log_dict({k: v for k, v in metrics.items()})
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        pred = self.forward(x, use_mask=False)
        self.test_metrics.update(pred, y)

        loss = nn.functional.mse_loss(pred, y)
        self.log(
            "test/loss", loss, prog_bar=True, sync_dist=True, batch_size=x.shape[0]
        )

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        lgr_logger.info("=== Pretrain Test Metrics ===")
        for k, v in metrics.items():
            lgr_logger.info(f"  {k}: {v.float():.4f}")
        lgr_logger.info("============================")
        self.log_dict({k: v for k, v in metrics.items()})
        self.test_metrics.reset()

    def predict_step(self, batch, batch_idx):
        # Expects batch to contain (x, timestamp) or just x
        # If batch is a tuple, assume it's (x, timestamp)
        if isinstance(batch, (list, tuple)):
            x, timestamps = batch[0], batch[1]
        else:
            x = batch
            timestamps = None

        embeddings = self.encode(x)

        # Store embeddings and timestamps
        # Convert to numpy for storage
        emb_np = embeddings.float().cpu().detach().numpy()

        if timestamps is not None:
            # Assuming timestamps are datetime or int
            # If they are tensors, convert to list or numpy
            if isinstance(timestamps, torch.Tensor):
                ts_np = timestamps.cpu().detach().numpy()
            else:
                ts_np = np.array(timestamps)
        else:
            ts_np = np.arange(len(emb_np))

        # Append to lists (be careful with memory usage for large datasets)
        self.pred_timestamps.append(ts_np)
        self.pred_embeddings.append(emb_np)

        return embeddings

    def on_predict_epoch_end(self, results):
        # Save embeddings to Zarr
        if self.save_embeddings_path:
            lgr_logger.info(f"Saving embeddings to {self.save_embeddings_path}")

            # Concatenate all batches
            all_timestamps = np.concatenate(self.pred_timestamps, axis=0)
            all_embeddings = np.concatenate(self.pred_embeddings, axis=0)

            # Create xarray Dataset
            # Embeddings shape: [total_samples, seq_len, embed_dim]
            # Or flatten to [total_samples, seq_len * embed_dim] if preferred

            # We need to be careful with dimensions.
            # Let's assume we keep it as [time, seq, dim]

            ds = xr.Dataset(
                {
                    "embeddings": (["timestep", "seq", "feature"], all_embeddings),
                },
                coords={
                    "timestep": all_timestamps,
                },
            )

            # Save to zarr
            # Using consolidated=True for faster loading later
            ds.to_zarr(self.save_embeddings_path, mode="w", consolidated=True)

            lgr_logger.info(
                f"Embeddings saved successfully. Shape: {all_embeddings.shape}"
            )

            # Clear buffers
            self.pred_timestamps = []
            self.pred_embeddings = []

    def configure_optimizers(self):
        optimizer_type = self.optimizer_dict.get("type", "adamw").lower()
        lr = self.optimizer_dict.get("lr", 1e-4)
        weight_decay = self.optimizer_dict.get("weight_decay", 0.0)
        eps = self.optimizer_dict.get("eps", 1e-8)
        betas = tuple(self.optimizer_dict.get("betas", [0.9, 0.999]))

        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                eps=eps,
                betas=betas,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                eps=eps,
                betas=betas,
            )

        scheduler_cfg = self.optimizer_dict.get("scheduler")
        if scheduler_cfg and scheduler_cfg.get("use") == "cosine_warmup":
            from torch.optim.lr_scheduler import (
                CosineAnnealingLR,
                LinearLR,
                SequentialLR,
            )

            total_steps = self.trainer.estimated_stepping_batches
            # Check for edge cases where Lightning returns infinity or valid steps are unknown
            if isinstance(total_steps, (float, int)) and (
                total_steps == float("inf") or total_steps == 0
            ):
                lgr_logger.warning(
                    "Warning: Could not calculate total steps automatically."
                )
                total_steps = scheduler_cfg["cosine_warmup"].get("total_steps", 10000)

            warmup_ratio = scheduler_cfg["cosine_warmup"].get("warmup_ratio", 0.1)
            min_lr = scheduler_cfg["cosine_warmup"].get("min_lr", 1e-6)

            warmup_steps = int(total_steps * warmup_ratio)
            train_steps = total_steps - warmup_steps

            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-6 / lr,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=train_steps,
                eta_min=min_lr,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return optimizer
