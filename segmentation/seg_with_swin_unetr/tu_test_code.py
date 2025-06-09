import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import torchio as tio
from torch.utils.data import random_split
import torch
from monai.transforms import (
    Compose,
    RandFlipd,
    # RandAffined,
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

# class TransformWrapper(Dataset):
#     def __init__(self, dataset, transforms):
#         self.dataset = dataset
#         self.transforms = transforms

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         sample = self.dataset[idx]
#         subject = tio.Subject(
#             image=tio.ScalarImage(tensor=sample['image']),
#             seg=tio.LabelMap(tensor=sample['seg']) if sample['seg'] is not None else None
#         )
#         subject = self.transforms(subject)
#         return {
#             'image': subject['image'].data,
#             'seg': subject['seg'].data if 'seg' in subject else None,
#         }

if __name__ == "__main__":

    image_dir = "/work/cuc.buithi/brats_challenge/data/train_flair_t1_t1ce_t2/imageTr"
    label_dir = "/work/cuc.buithi/brats_challenge/data/train_flair_t1_t1ce_t2/labelTr"

     ########################## TRANSFORM ####################################

    train_transforms = Compose([
        RandFlipd(keys=["image", "seg"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "seg"], spatial_axis=1, prob=0.5),
        # RandAffined(keys=["image", "seg"], prob=0.5, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1)),
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

    ################ MODEL ##############################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = SwinUNETR(
        img_size=(160, 160, 96),
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        feature_size=48,        
        norm_name='instance',
        use_checkpoint=False
    ).to(device)


    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Initialize metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Move model to device immediately after definition or loading
    model = model.to(device)

    epochs = 1  

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader):
            images = batch['image'].to(device)  # [B, C, D, H, W]
            labels = batch['seg'].to(device).long()  # [B, 1, D, H, W]
        
            optimizer.zero_grad()
            outputs = model(images)  # [B, C, D, H, W]
            loss = loss_function(outputs, labels)  # Safe, MONAI will internally one-hot from [B, 1, D, H, W] to [B, C, D, H, W]
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} average loss: {epoch_loss / len(train_loader):.4f}")

        # --- Validation ---
        model.eval()
        dice_metric.reset()

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['seg'].to(device).long()  # Should be [B, 1, D, H, W]
        
                outputs = model(images)
        
                # Ensure same size (safe crop to min size in all dims)
                min_d = min(outputs.shape[2], labels.shape[2])
                min_h = min(outputs.shape[3], labels.shape[3])
                min_w = min(outputs.shape[4], labels.shape[4])
        
                outputs = outputs[:, :, :min_d, :min_h, :min_w]
                labels = labels[:, :, :min_d, :min_h, :min_w]
        
                # Safe one-hot
                labels_one_hot = one_hot(labels, num_classes=outputs.shape[1])
        
                preds = torch.argmax(outputs, dim=1, keepdim=True)  # [B, 1, D, H, W]
                preds_one_hot = one_hot(preds, num_classes=outputs.shape[1])
                
                dice_metric(y_pred=preds_one_hot, y=labels_one_hot)
        # Aggregate and print metrics
        dice_score = dice_metric.aggregate().item()

        print(f"Epoch {epoch+1} validation Dice: {dice_score:.4f}")
