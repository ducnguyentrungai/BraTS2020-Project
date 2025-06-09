import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import random_split
import torch
import pytorch_lightning as pl
from monai.transforms import (
    Compose,
    RandFlipd,
    RandAffined,
    Rand3DElasticd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    NormalizeIntensityd,
    SpatialPadd,
    CenterSpatialCropd,
)
from torch.utils.data import Dataset
from monai.transforms import LoadImage
from torch.utils.data import DataLoader, Dataset

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR

from tqdm import tqdm
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.networks.utils import one_hot

import warnings
warnings.filterwarnings("ignore")

loader = LoadImage(image_only=True)  # Chỉ load ảnh, không metadata

class MRI_Dataset(Dataset):
    def __init__(self, image_dir, label_dir=None, transforms=None, mode='train'):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.file_list = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith('.nii.gz')
        ])
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        image_path = os.path.join(self.image_dir, file_name)

        # Load image numpy array
        img = loader(image_path)

        # Chuyển shape về (4, H, W, D) nếu cần
        if img.ndim == 4 and img.shape[-1] == 4:
            img = np.moveaxis(img, -1, 0)

        img = img.astype(np.float32)

        sample = {'image': img}

        # Nếu mode train và label_dir được cung cấp thì load segmentation
        if self.mode == 'train' and self.label_dir is not None:
            # Giả sử label file tên giống file ảnh
            seg_path = os.path.join(self.label_dir, file_name)
            if os.path.exists(seg_path):
                seg = loader(seg_path).astype(np.uint8)
                # Nếu segmentation có 3D shape (H,W,D) thì mở rộng thành (1,H,W,D)
                if seg.ndim == 3:
                    seg = np.expand_dims(seg, axis=0)
                # Nếu segmentation có 4D và last dim là 1 thì chuyển về (1,H,W,D)
                elif seg.ndim == 4 and seg.shape[-1] == 1:
                    seg = np.moveaxis(seg, -1, 0)
                # Remap label nếu cần (ví dụ bệnh BraTS)
                seg_remap = np.zeros_like(seg, dtype=np.uint8)
                seg_remap[seg == 1] = 0
                seg_remap[seg == 2] = 1
                seg_remap[seg == 4] = 2
                sample['seg'] = seg_remap
            else:
                # Nếu không tìm thấy segmentation, tạo dummy mask zeros
                dummy_seg = np.zeros(img.shape[1:], dtype=np.uint8)
                dummy_seg = np.expand_dims(dummy_seg, axis=0)
                sample['seg'] = dummy_seg
        else:
            # Nếu không có label_dir hoặc mode khác train thì set seg = None
            sample['seg'] = None

        # Áp dụng transform
        if self.transforms:
            # Nếu seg = None mà transform có key "seg" thì tạo dummy mask để tránh lỗi transform
            if sample['seg'] is None:
                dummy_seg = np.zeros(img.shape[1:], dtype=np.uint8)
                dummy_seg = np.expand_dims(dummy_seg, axis=0)
                sample['seg'] = dummy_seg
                sample = self.transforms(sample)
                sample['seg'] = None  # Loại bỏ dummy seg sau transform
            else:
                sample = self.transforms(sample)

        return {
            'image': sample['image'],
            'seg': sample['seg']
        }


class TransformWrapper(Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms  # MONAI transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]  # sample dict: {'image': tensor or ndarray, 'seg': tensor or None}

        # Nếu transforms có, áp dụng trực tiếp
        if self.transforms:
            # Lưu ý: MONAI transforms expect dict with keys matching transforms
            sample = self.transforms(sample)

        return {
            'image': sample['image'],
            'seg': sample['seg'] if 'seg' in sample else None,
        }



###################### PYTORCH LIGHTNING ############################
class SwinUNETRLightning(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = SwinUNETR(
            img_size=(160, 160, 96),
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            feature_size=48,        
            norm_name='instance',
            use_checkpoint=False
        )
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.lr = lr
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['seg'].long()
        outputs = self(images)
        loss = self.loss_function(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['seg'].long()
        outputs = self(images)

        # min_d = min(outputs.shape[2], labels.shape[2])
        # min_h = min(outputs.shape[3], labels.shape[3])
        # min_w = min(outputs.shape[4], labels.shape[4])

        # outputs = outputs[:, :, :min_d, :min_h, :min_w]
        # labels = labels[:, :, :min_d, :min_h, :min_w]

        preds = torch.argmax(outputs, dim=1, keepdim=True)  # [B, 1, D, H, W]
        
        # ➕ chuyển sang one-hot
        preds = one_hot(preds, num_classes=outputs.shape[1])
        labels = one_hot(labels, num_classes=outputs.shape[1])

        self.dice_metric(y_pred=preds, y=labels)


    def on_validation_epoch_end(self):
        dice_score = self.dice_metric.aggregate().item()
        self.log('val_dice', dice_score, prog_bar=True)
        self.dice_metric.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

class BrainSegDataModule(pl.LightningDataModule):
    def __init__(self, image_dir, label_dir, batch_size=6, num_workers=2):
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transforms = Compose([
            RandFlipd(keys=["image", "seg"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "seg"], spatial_axis=1, prob=0.5),
            RandAffined(keys=["image", "seg"], prob=0.5, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1)),
            Rand3DElasticd(keys=["image", "seg"], sigma_range=(5, 7), magnitude_range=(100, 200), prob=0.3),
            RandGaussianNoised(keys=["image"], prob=0.2),
            RandGaussianSmoothd(keys=["image"], prob=0.3),
            NormalizeIntensityd(keys=["image"]),
            SpatialPadd(keys=["image", "seg"], spatial_size=(96, 96, 96)),
            CenterSpatialCropd(keys=["image", "seg"], roi_size=(96, 96, 96)),
        ])

        self.test_transforms = Compose([
            NormalizeIntensityd(keys=["image"]),
            SpatialPadd(keys=["image", "seg"], spatial_size=(96, 96, 96)),
            CenterSpatialCropd(keys=["image", "seg"], roi_size=(96, 96, 96)),
        ])

    def setup(self, stage=None):
        full_dataset = MRI_Dataset(self.image_dir, label_dir=self.label_dir, transforms=None, mode='train')

        train_size = int(0.89* len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        train_split, val_split, test_split = random_split(full_dataset, [train_size, val_size, test_size])

        self.train_dataset = TransformWrapper(train_split, self.train_transforms)
        self.val_dataset = TransformWrapper(val_split, self.test_transforms)
        self.test_dataset = TransformWrapper(test_split, self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

# def auto_select_gpus(n=3, threshold_mem_mb=800, threshold_util=10):
#     try:
#         nvmlInit()
#         device_count = nvmlDeviceGetCount()
#         gpus = []

#         for i in range(device_count):
#             handle = nvmlDeviceGetHandleByIndex(i)
#             mem_info = nvmlDeviceGetMemoryInfo(handle)
#             util_info = nvmlDeviceGetUtilizationRates(handle)

#             mem_used_mb = mem_info.used // 1024**2
#             gpu_util = util_info.gpu

#             if mem_used_mb < threshold_mem_mb and gpu_util < threshold_util:
#                 gpus.append((i, mem_used_mb, gpu_util))

#         if not gpus:
#             return None

#         gpus.sort(key=lambda x: (x[1], x[2]))
#         selected_ids = [g[0] for g in gpus[:n]]
#         return selected_ids

#     except Exception as e:
#         print(f"GPU auto-selection failed: {e}")
#         return None
#     finally:
#         try:
#             nvmlShutdown()
#         except:
#             pass

if __name__ == "__main__":

    image_dir = "/work/cuc.buithi/brats_challenge/data/train_flair_t1_t1ce_t2/imageTr"
    label_dir = "/work/cuc.buithi/brats_challenge/data/train_flair_t1_t1ce_t2/labelTr"

     ########################## TRANSFORM ####################################

    train_transforms = Compose([
        RandFlipd(keys=["image", "seg"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "seg"], spatial_axis=1, prob=0.5),
        RandAffined(keys=["image", "seg"], prob=0.5, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1)),
        Rand3DElasticd(keys=["image", "seg"], sigma_range=(5, 7), magnitude_range=(100, 200), prob=0.3),
        RandGaussianNoised(keys=["image"], prob=0.2),
        RandGaussianSmoothd(keys=["image"], prob=0.3),
        NormalizeIntensityd(keys=["image"]),
        SpatialPadd(keys=["image", "seg"], spatial_size=(96, 96, 96)),
        CenterSpatialCropd(keys=["image", "seg"], roi_size=(96, 96, 96)),
    ])

    test_transforms = Compose([
        NormalizeIntensityd(keys=["image"]),
        SpatialPadd(keys=["image", "seg"], spatial_size=(96, 96, 96)),
        CenterSpatialCropd(keys=["image", "seg"], roi_size=(96, 96, 96)),
    ])

    ########################## TRANSFORM ####################################

    dataset = MRI_Dataset(image_dir, label_dir=label_dir, transforms=train_transforms, mode='train')

    print("Dataset length:", len(dataset))

    sample = dataset[0]
    print("Image shape:", sample['image'].shape)
    if sample['seg'] is not None:
        print("Segmentation shape:", sample['seg'].shape)
    else:
        print("Segmentation is None")

    ######################### Split data ##############################

    print("######################### Split data ##############################")
    full_dataset = MRI_Dataset(image_dir, label_dir=label_dir, transforms=train_transforms, mode='train')

    train_size = int(0.89 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    # train_split, val_split = random_split(full_dataset, [train_size, val_size])
    train_split, val_split, test_split = random_split(full_dataset, [train_size, val_size, test_size])

    print(f"Số lượng mẫu trong train_split: {len(train_split)}")
    print(f"Số lượng mẫu trong val_split: {len(val_split)}")
    print(f"Số lượng mẫu trong test_split: {len(test_split)}")

          
    ######################### Split data ##############################

    train_dataset = TransformWrapper(train_split, train_transforms)
    val_dataset = TransformWrapper(val_split, test_transforms)
    test_dataset = TransformWrapper(test_split, test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=2) #4GPU=8batch, 3GPU=6batch, 
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)

    ################ PYTROCH LIGHTNING #################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    datamodule = BrainSegDataModule(image_dir, label_dir, batch_size=6, num_workers=2)
    model = SwinUNETRLightning(lr=1e-4)

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="gpu",
        devices=[0, 3],  # sử dụng tất cả GPU
        precision=16,  # mixed precision (optional)
        log_every_n_steps=5,
    )

    trainer.fit(model, datamodule=datamodule)

    ################ PYTORCH LIGHTNING #################

    ################ MODEL ##############################
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)

    # model = SwinUNETR(
    #     img_size=(160, 160, 96),
    #     spatial_dims=3,
    #     in_channels=4,
    #     out_channels=3,
    #     feature_size=48,        
    #     norm_name='instance',
    #     use_checkpoint=False
    # ).to(device)


    # loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # # Initialize metrics
    # dice_metric = DiceMetric(include_background=False, reduction="mean")

    # # Move model to device immediately after definition or loading
    # model = model.to(device)

    # epochs = 1  

    # for epoch in range(epochs):
    #     model.train()
    #     epoch_loss = 0
    #     for batch in tqdm(train_loader):
    #         images = batch['image'].to(device)  # [B, C, D, H, W]
    #         labels = batch['seg'].to(device).long()  # [B, 1, D, H, W]
        
    #         optimizer.zero_grad()
    #         outputs = model(images)  # [B, C, D, H, W]
    #         loss = loss_function(outputs, labels)  # Safe, MONAI will internally one-hot from [B, 1, D, H, W] to [B, C, D, H, W]
    #         loss.backward()
    #         optimizer.step()

    #         epoch_loss += loss.item()

    #     print(f"Epoch {epoch+1} average loss: {epoch_loss / len(train_loader):.4f}")

    #     # --- Validation ---
    #     model.eval()
    #     dice_metric.reset()

    #     with torch.no_grad():
    #         for batch in val_loader:
    #             images = batch['image'].to(device)
    #             labels = batch['seg'].to(device).long()  # Should be [B, 1, D, H, W]
        
    #             outputs = model(images)
        
    #             # Ensure same size (safe crop to min size in all dims)
    #             min_d = min(outputs.shape[2], labels.shape[2])
    #             min_h = min(outputs.shape[3], labels.shape[3])
    #             min_w = min(outputs.shape[4], labels.shape[4])
        
    #             outputs = outputs[:, :, :min_d, :min_h, :min_w]
    #             labels = labels[:, :, :min_d, :min_h, :min_w]
        
    #             # Safe one-hot
    #             labels_one_hot = one_hot(labels, num_classes=outputs.shape[1])
        
    #             preds = torch.argmax(outputs, dim=1, keepdim=True)  # [B, 1, D, H, W]
    #             preds_one_hot = one_hot(preds, num_classes=outputs.shape[1])
                
    #             dice_metric(y_pred=preds_one_hot, y=labels_one_hot)
    #     # Aggregate and print metrics
    #     dice_score = dice_metric.aggregate().item()

    #     print(f"Epoch {epoch+1} validation Dice: {dice_score:.4f}")
