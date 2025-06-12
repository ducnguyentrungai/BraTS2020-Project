import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRanged, Resized, ToTensord,
    NormalizeIntensityd
)

# ===== Thêm đường dẫn thư mục cha để import module =====
current_dir = os.path.dirname(os.path.abspath(__file__))  # /.../visualization
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))  # /.../seg_with_swin_unetr
sys.path.append(parent_dir)

from lightning_module import LitSegSwinUNETR
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss

if __name__ == "__main__":
    # === Bước 1: Load mô hình từ checkpoint ===
    swin_model = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=4,
        feature_size=48,
        norm_name='batch',
        use_checkpoint=True,
        use_v2=True
    )

    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)

    ckpt_path = "/work/cuc.buithi/brats_challenge/code/segmentation/seg_with_swin_unetr/swin_unetr_v2/checkpoints/best_model-epoch=29-val_dice=0.6091.ckpt"
    model = LitSegSwinUNETR.load_from_checkpoint(
        ckpt_path,
        model=swin_model,
        loss_fn=loss_fn,
        lr=2e-4,
        optim='adamw',
        out_path="logs",
        num_classes=4,
        include_background=False
    )
    model.eval()
    swin_model = model.model

    # === Bước 2: Load và xử lý ảnh ===
    image_path = "/work/cuc.buithi/brats_challenge/data/train_t1_t1ce_t2_flair/imageTr/image_327.nii.gz"
    label_path = "/work/cuc.buithi/brats_challenge/data/train_t1_t1ce_t2_flair/labelTr/label_327.nii.gz"

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),       # Z-score normalization
        Resized(keys=["image", "label"], spatial_size=(128, 128, 128), mode=("trilinear", "nearest")),
        ToTensord(keys=["image", "label"]),
    ])

    sample = {"image": image_path, "label": label_path}
    data = val_transforms(sample)
    image_tensor = data["image"].unsqueeze(0)  # (1, 4, D, H, W)
    label_tensor = data["label"]  # (1, D, H, W)

    # === Bước 3: Dự đoán segmentation ===
    with torch.no_grad():
        pred = swin_model(image_tensor)  # (1, 4, D, H, W)
        pred_seg = torch.argmax(pred, dim=1).squeeze(0).cpu()  # (D, H, W)

    # === Bước 4: Hiển thị lát cắt giữa ===
    slice_index = pred_seg.shape[0] // 2
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image_tensor[0, 0, slice_index], cmap='gray')
    ax[0].set_title(f"FLAIR (slice {slice_index})")
    ax[0].axis('off')

    ax[1].imshow(label_tensor[0, slice_index], cmap='nipy_spectral')
    ax[1].set_title("Ground Truth")
    ax[1].axis('off')

    ax[2].imshow(pred_seg[slice_index], cmap='nipy_spectral')
    ax[2].set_title("Predicted Mask")
    ax[2].axis('off')

    plt.tight_layout()
    plt.savefig('check.png')
    plt.show()
