import os
import glob
import pandas as pd
from typing import List, Union, Optional, Callable
from monai.data import CacheDataset, list_data_collate, partition_dataset
from torch.utils.data import DataLoader
import torch.distributed as dist
from pytorch_lightning import LightningDataModule
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ToTensord, MapTransform


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.io_utils import create_data_dicts_from_bratsid

def is_rank_zero():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True

def rank_zero_print(*args, **kwargs):
    if is_rank_zero():
        print(*args, **kwargs)

class BratsDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 table_path: str,
                 spatial_size: Union[List[int], int] = (128, 128, 128),
                 batch_size: int = 4,
                 num_workers: int = 2,
                 train_percent: float = 0.9,
                 modalities: List[str] = ['t1', 't1ce', 't2', 'flair'],
                 transform_fn: Optional[Callable[[bool], Compose]] = None):
        super().__init__()
        self.data_dir = data_dir
        self.table_path = table_path
        self.spatial_size = spatial_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_percent = train_percent
        self.modalities = modalities
        self.transform_fn = transform_fn

    def setup(self, stage: Optional[str] = None):
        all_cases = sorted(glob.glob(os.path.join(self.data_dir, "BraTS20_Training_*")))
        rank_zero_print(f"Found {len(all_cases)} folders.")

        train_dicts, val_dicts= create_data_dicts_from_bratsid(
            root_dir=self.data_dir,
            table_path=self.table_path,
            modalities=self.modalities,
            train_percent=self.train_percent
        )

        rank_zero_print(f"Train samples: {len(train_dicts)}, Val samples: {len(val_dicts)}")

        # train_dicts = train_dicts[:8]
        # val_dicts = val_dicts[:4]

        if dist.is_available() and dist.is_initialized():
            
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            train_partitioned = partition_dataset(
                data=train_dicts,
                num_partitions=world_size,
                shuffle=True,
                even_divisible=True
            )
            train_dicts = train_partitioned[rank]

        if stage in ("fit", None):
            self.train_dataset = CacheDataset(
                data=train_dicts,
                transform=self.transform_fn(is_train=True),
                cache_rate=1.0,
                num_workers=self.num_workers,
            )
            self.val_dataset = CacheDataset(
                data=val_dicts,
                transform=self.transform_fn(is_train=False),
                cache_rate=1.0,
                num_workers=self.num_workers,
            )

        elif stage in ("test", "predict"):
            self.val_dataset = CacheDataset(
                data=val_dicts,
                transform=self.transform_fn(is_train=False),
                cache_rate=1.0,
                num_workers=self.num_workers,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=list_data_collate,
        )

    def val_dataloader(self):
        return self._shared_eval_dataloader()

    def test_dataloader(self):
        return self._shared_eval_dataloader()

    def predict_dataloader(self):
        return self._shared_eval_dataloader()

    def _shared_eval_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=list_data_collate,
        )

