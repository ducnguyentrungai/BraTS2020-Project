import torch
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import sys
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityd, ToTensord, ScaleIntensityRanged,
    Resized, 
)
# Thêm thư mục cha vào PYTHONPATH để import được lightning_module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from lightning_module import LitSegSwinUNETR  # thay bằng tên thật của bạn

# === Bước 1: Load mô hình từ checkpoint ===
ckpt_path = "/work/cuc.buithi/brats_challenge/code/segmentation/seg_with_swin_unetr/checkpoints/best_model-epoch=470-val_dice=0.7113.ckpt"
model = LitSegSwinUNETR.load_from_checkpoint(ckpt_path)
model.eval()

# Nếu cần mô hình gốc để predict trực tiếp:
swin_model = model.model  # hoặc .unetr nếu bạn đặt tên khác

# === Bước 2: Load và xử lý ảnh ===
image_path = '/work/cuc.buithi/brats_challenge/data/train_flair_t1_t1ce_t2/imageTr/image_00000.nii.gz'
label_path = '/work/cuc.buithi/brats_challenge/data/train_flair_t1_t1ce_t2/labelTr/label_00000.nii.gz'

val_transforms = Compose([
  LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=["image", "label"], spatial_size=(128, 128, 128), mode=("trilinear", "nearest")),
    ToTensord(keys=["image", "label"])
])

# Tạo input dict
sample = {"image": image_path, "label": label_path}
data = val_transforms(sample)
image_tensor = data["image"].unsqueeze(0)  # (1, 4, D, H, W)
label_tensor = data["label"]  # (1, D, H, W) hoặc (D, H, W)

# === Bước 3: Dự đoán segmentation ===
with torch.no_grad():
    pred = swin_model(image_tensor)  # shape (1, 4, D, H, W)
    pred_seg = torch.argmax(pred, dim=1).squeeze(0).cpu()  # (D, H, W)

# === Bước 4: Hiển thị lát cắt giữa ===
slice_index = pred_seg.shape[0] // 2
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# ảnh gốc - flair (channel 0)
ax[0].imshow(image_tensor[0, 0, slice_index], cmap='gray')
ax[0].set_title("FLAIR (slice {})".format(slice_index))
ax[0].axis('off')

# ground truth
ax[1].imshow(label_tensor[0, slice_index], cmap='nipy_spectral')
ax[1].set_title("Ground Truth")
ax[1].axis('off')

# dự đoán
ax[2].imshow(pred_seg[slice_index], cmap='nipy_spectral')
ax[2].set_title("Predicted Mask")
ax[2].axis('off')

plt.tight_layout()
plt.savefig('test.png')
plt.show()
