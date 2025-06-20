import os
import glob
from typing import List, Optional, Callable, Sequence, Union
import torch.distributed as dist
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from monai.data import CacheDataset, list_data_collate, partition_dataset, PersistentDataset
from monai.transforms import Compose
from sklearn.model_selection import train_test_split
def is_rank_zero():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True

def rank_zero_print(*args, **kwargs):
    if is_rank_zero():
        print(*args, **kwargs)


def create_data_dicts(data_dirs: List[str], modalities: List[str]) -> List[dict]:
    """
    Tạo danh sách dict chuẩn MONAI: {'image': [...], 'label': ...}
    """
    data_dicts = []
    for case_dir in data_dirs:
        base = os.path.basename(case_dir)
        images = [os.path.join(case_dir, f"{base}_{mod}.nii.gz") for mod in modalities]
        label = os.path.join(case_dir, f"{base}_seg.nii.gz")
        if all(os.path.exists(p) for p in images) and os.path.exists(label):
            data_dicts.append({"image": images, "label": label})
    return data_dicts


class BratsDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        spatial_size: Union[Sequence[int], int] = (128, 128, 128),
        batch_size: int = 2,
        num_workers: int = 2,
        train_percent: float = 0.825,
        modalities: List[str] = ['t1', 't1ce', 't2', 'flair'],
        transform_fn: Optional[Callable[[bool], Compose]] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.spatial_size = spatial_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_percent = train_percent
        self.modalities = modalities
        self.transform_fn = transform_fn

    def setup(self, stage: Optional[str] = None):
        all_cases = sorted(glob.glob(os.path.join(self.data_dir, "BraTS2021_*")))
        rank_zero_print(f"Found {len(all_cases)} cases.")

        train_cases, val_cases = train_test_split(all_cases, train_size=self.train_percent, random_state=42, shuffle=True)
        train_dicts = create_data_dicts(train_cases, self.modalities)
        val_dicts = create_data_dicts(val_cases, self.modalities)

        # ✅ Khai báo thư mục cache
        train_cache_dir = os.path.join(self.data_dir, "train_cache")
        val_cache_dir = os.path.join(self.data_dir, "val_cache")
        os.makedirs(train_cache_dir, exist_ok=True)
        os.makedirs(val_cache_dir, exist_ok=True)

        if stage in ("fit", None):
            self.train_dataset = PersistentDataset(
                data=train_dicts,
                transform=self.transform_fn(is_train=True),
                cache_dir=train_cache_dir,
            )

            self.val_dataset = PersistentDataset(
                data=val_dicts,
                transform=self.transform_fn(is_train=False),
                cache_dir=val_cache_dir,
            )

        elif stage in ("test", "predict"):
            self.val_dataset = PersistentDataset(
                data=val_dicts,
                transform=self.transform_fn(is_train=False),
                cache_dir=val_cache_dir,
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
        
        
