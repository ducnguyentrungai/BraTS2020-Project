import os
import json
import glob
from typing import List, Optional, Callable, Sequence, Union
import torch.distributed as dist
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from monai.data import CacheDataset, list_data_collate, partition_dataset, PersistentDataset, SmartCacheDataset
from monai.transforms import Compose
from sklearn.model_selection import train_test_split
import nibabel as nib
def is_rank_zero():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True

def rank_zero_print(*args, **kwargs):
    if is_rank_zero():
        print(*args, **kwargs)


# def create_data_dicts(data_dirs: List[str], modalities: List[str]) -> List[dict]:
#     """
#     Tạo danh sách dict chuẩn MONAI: {'image': [...], 'label': ...}
#     """
#     data_dicts = []
#     for case_dir in data_dirs:
#         base = os.path.basename(case_dir)
#         images = [os.path.join(case_dir, f"{base}_{mod}.nii.gz") for mod in modalities]
#         label = os.path.join(case_dir, f"{base}_seg.nii.gz")
#         if all(os.path.exists(p) for p in images) and os.path.exists(label):
#             data_dicts.append({"image": images, "label": label})
#     return data_dicts


def create_data_dicts(
    data_dirs: List[str],
    modalities: List[str],
    error_json_path: str = None,
    expected_dim: int = 3  # mỗi ảnh đơn modal phải có shape [H, W, D]
) -> List[dict]:
    data_dicts = []
    error_cases = []

    for case_dir in data_dirs:
        base = os.path.basename(case_dir)
        images = [os.path.join(case_dir, f"{base}_{mod}.nii.gz") for mod in modalities]
        label = os.path.join(case_dir, f"{base}_seg.nii.gz")

        case_error = {"case_id": base}
        missing = [p for p in images if not os.path.exists(p)]
        if not os.path.exists(label):
            missing.append(label)

        if missing:
            case_error["missing_files"] = missing
        else:
            bad_shapes = []
            for i, img_path in enumerate(images):
                try:
                    img = nib.load(img_path)
                    if len(img.shape) != expected_dim:
                        bad_shapes.append({
                            "modality": modalities[i],
                            "shape": img.shape,
                            "path": img_path
                        })
                except Exception as e:
                    bad_shapes.append({
                        "modality": modalities[i],
                        "error": str(e),
                        "path": img_path
                    })

            if bad_shapes:
                case_error["bad_shapes"] = bad_shapes

        if "missing_files" in case_error or "bad_shapes" in case_error:
            error_cases.append(case_error)
        else:
            data_dicts.append({
                "image": images,     # sẽ được LoadImaged + ConcatItemsd
                "label": label,
                "case_id": base
            })

    if error_json_path and error_cases:
        with open(error_json_path, "w") as f:
            json.dump(error_cases, f, indent=2)
        print(f"❌ Ghi {len(error_cases)} case lỗi vào: {error_json_path}")
    else:
        print("✅ Không có case lỗi.")

    print(f"✅ Tổng số case hợp lệ: {len(data_dicts)} / {len(data_dirs)}")
    return data_dicts


def create_data_dicts(
    data_dirs: List[str],
    modalities: List[str],
    error_json_path: str = None,
    expected_dim: int = 3,  # số chiều không gian kỳ vọng: thường là 3D (H, W, D)
) -> List[dict]:
    data_dicts = []
    error_cases = []

    for case_dir in data_dirs:
        base = os.path.basename(case_dir)
        images = [os.path.join(case_dir, f"{base}_{mod}.nii.gz") for mod in modalities]
        label = os.path.join(case_dir, f"{base}_seg.nii.gz")

        case_error = {"case_id": base}
        missing = [p for p in images if not os.path.exists(p)]
        if not os.path.exists(label):
            missing.append(label)

        if missing:
            case_error["missing_files"] = missing
        else:
            bad_shapes = []
            for i, img_path in enumerate(images):
                try:
                    img = nib.load(img_path)
                    if len(img.shape) != expected_dim:
                        bad_shapes.append({
                            "modality": modalities[i],
                            "shape": img.shape,
                            "path": img_path
                        })
                except Exception as e:
                    bad_shapes.append({
                        "modality": modalities[i],
                        "error": str(e),
                        "path": img_path
                    })
            if bad_shapes:
                case_error["bad_shapes"] = bad_shapes

        if "missing_files" in case_error or "bad_shapes" in case_error:
            error_cases.append(case_error)
        else:
            data_dicts.append({
                "image": images,
                "label": label,
            })

    if error_json_path and error_cases:
        with open(error_json_path, "w") as f:
            json.dump(error_cases, f, indent=2)
        print(f"❌ Ghi {len(error_cases)} case lỗi vào: {error_json_path}")
    else:
        print("✅ Không có case lỗi.")

    print(f"✅ Tổng số case hợp lệ: {len(data_dicts)} / {len(data_dirs)}")
    return data_dicts



def setup_case_splits(data_dir, modalities, train_percent=0.825, debug_limit=None):
    all_cases = sorted(glob.glob(os.path.join(data_dir, "BraTS2021_*")))
    rank_zero_print(f"🔍 Tổng số case tìm thấy: {len(all_cases)}")

    train_cases, val_cases = train_test_split(
        all_cases, train_size=train_percent, shuffle=True, random_state=42
    )
    rank_zero_print(f"📂 Train: {len(train_cases)} | Val: {len(val_cases)}")

    train_dicts = create_data_dicts(
        train_cases, modalities, error_json_path="train_missing.json"
    )
    val_dicts = create_data_dicts(
        val_cases, modalities, error_json_path="val_missing.json"
    )

    # Nếu cần debug nhanh
    if debug_limit:
        train_dicts = train_dicts[:debug_limit]
        val_dicts = val_dicts[:debug_limit // 2]
        rank_zero_print(f"🧪 Debug mode: train={len(train_dicts)}, val={len(val_dicts)}")

    return train_dicts, val_dicts


class BratsDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        spatial_size: Union[Sequence[int], int] = (128, 128, 128),
        batch_size: int = 2,
        num_workers: int = 2,
        train_percent: float = 0.83,
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
        train_dicts, val_dicts =  setup_case_splits(self.data_dir, modalities=self.modalities, train_percent=self.train_percent)

        # Chia dữ liệu giữa các GPU nếu dùng DDP
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
            
            # === Load data with CacheDataset === 
            # self.train_dataset = CacheDataset(
            #     data=train_dicts,
            #     transform=self.transform_fn(is_train=True),
            #     cache_rate=1.0,
            #     num_workers=self.num_workers,
            # )
            # self.val_dataset = CacheDataset(
            #     data=val_dicts,
            #     transform=self.transform_fn(is_train=False),
            #     cache_rate=1.0,
            #     num_workers=self.num_workers,
            # )
            
            # === Load data with SmartCacheDataset ===
            self.train_dataset = SmartCacheDataset(
                data=train_dicts,
                transform=self.transform_fn(is_train=True),
                cache_num=256,         # số lượng sample cache ban đầu, bạn có thể điều chỉnh
                replace_rate=0.3,      # tỷ lệ sample được thay thế mỗi epoch
                num_init_workers = self.num_workers
            )
            
            self.val_dataset = SmartCacheDataset(
                data=val_dicts,
                transform=self.transform_fn(is_train=False),
                cache_num= len(val_dicts),       
                replace_rate=1.0,
                num_init_workers = self.num_workers
            )

        elif stage in ("test", "predict"):
            # self.val_dataset = CacheDataset(
            #     data=val_dicts,
            #     transform=self.transform_fn(is_train=False),
            #     cache_rate=1.0,
            #     num_workers=self.num_workers,
            # )
            
            self.val_dataset = SmartCacheDataset(
                data=val_dicts,
                transform=self.transform_fn(is_train=True),
                cache_num= len(val_dicts),         # số lượng sample cache ban đầu, bạn có thể điều chỉnh
                replace_rate=1.0,      # tỷ lệ sample được thay thế mỗi epoch
                num_init_workers = self.num_workers
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
        
        
