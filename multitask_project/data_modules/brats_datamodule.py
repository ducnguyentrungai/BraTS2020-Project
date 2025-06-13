import os
import glob
from typing import Optional, List
from monai.data import CacheDataset, DataLoader
from pytorch_lightning import LightningDataModule
from data_modules.data_process import get_transforms, extract_data_dicts

class BraTSDatasetModule(LightningDataModule):
    def __init__(self, data_path: str,
                 tabular_path: str,
                 train_percent: float=0.8,
                 batch_size: int=2,
                 num_workers: int=2,
                 cache_num: int=100,
                 replace_rate: float=0.1,
                 ):
        
        super().__init__()
        self.train_files, self.test_files = extract_data_dicts(data_dir=data_path, train_percent=train_percent)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.replace_rate = replace_rate
        self.cache_num = cache_num
        
    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = CacheDataset(
                data=self.train_files,
                transform=get_transforms(is_train=True),
                cache_rate=self.cache_rate,
                num_workers=self.num_workers
            )
        if stage in (None, "test"):
            self.test_dataset = CacheDataset(
                data=self.test_files,
                transform=get_transforms(is_train=False),
                cache_rate=self.cache_rate,
                num_workers=self.num_workers
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.num_workers,
            pin_memory = True
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size = self.batch_size, 
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = True
        )
  
    