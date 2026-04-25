import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from flare_surya.dataset.solar_pretrain_dataset import SolarPretrainDataset


class SolarPretrainDataModule(pl.LightningDataModule):
    """
    A DataModule for self-supervised pre-training of solar data.
    """

    def __init__(
        self,
        zarr_path: str,
        train_index_path: str,
        val_index_path: str,
        test_index_path: str,
        channels: list[str],
        batch_size: int = 32,
        num_workers: int = 4,
        data_type: str = "1d",
        scalers: DictConfig | None = None,
    ):
        """
        Args:
            zarr_path (str): Path to the Zarr store.
            train_index_path (str): Path to the training index CSV.
            val_index_path (str): Path to the validation index CSV.
            test_index_path (str): Path to the test index CSV.
            channels (list[str]): List of channels to load.
            batch_size (int): Batch size for training.
            num_workers (int): Number of workers for data loading.
            data_type (str): Type of data, either '1d' or '2d'.
            scalers (DictConfig, optional): Normalization statistics.
        """
        super().__init__()
        self.zarr_path = zarr_path
        self.train_index_path = train_index_path
        self.val_index_path = val_index_path
        self.test_index_path = test_index_path
        self.channels = channels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_type = data_type
        self.scalers = OmegaConf.load(scalers)

    def setup(self, stage=None):
        """
        Load datasets.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = SolarPretrainDataset(
                zarr_path=self.zarr_path,
                index_path=self.train_index_path,
                channels=self.channels,
                scalers=self.scalers,
                data_type=self.data_type,
                phase="train",
            )
            self.val_dataset = SolarPretrainDataset(
                zarr_path=self.zarr_path,
                index_path=self.val_index_path,
                channels=self.channels,
                scalers=self.scalers,
                data_type=self.data_type,
                phase="val",
            )

        if stage == "test" or stage is None:
            self.test_dataset = SolarPretrainDataset(
                zarr_path=self.zarr_path,
                index_path=self.test_index_path,
                channels=self.channels,
                scalers=self.scalers,
                data_type=self.data_type,
                phase="test",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
